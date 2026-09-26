"""Docling OCR engine that reads scanned pages in horizontal bands.

Why: Apple Vision (ocrmac) silently downscales very large images, so a full
newspaper page OCR'd in one go comes back garbled. Full-width bands ~800px tall
come back near-perfect. Bands overlap so no text line is only half inside one;
the duplicate lines this creates are removed before Docling builds the layout.
"""

from typing import ClassVar, Literal

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrMacOptions
from docling.models.factories import get_ocr_factory
from docling.models.stages.ocr.ocr_mac_model import OcrMacModel
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import TextCell


class BandedOcrMacOptions(OcrMacOptions):
    kind: ClassVar[Literal["ocrmac_banded"]] = "ocrmac_banded"
    band_height: float = 180.0  # in PDF points; x scale 4 = ~720px bands
    band_overlap: float = 60.0  # taller than a headline line, so every line is whole in some band


class BandedOcrMacModel(OcrMacModel):
    options: BandedOcrMacOptions

    @classmethod
    def get_options_type(cls):
        return BandedOcrMacOptions

    def get_ocr_rects(self, page: Page) -> list[BoundingBox]:
        width, height = page.size.width, page.size.height
        step = self.options.band_height - self.options.band_overlap
        rects, top = [], 0.0
        while top < height:
            bottom = min(top + self.options.band_height, height)
            rects.append(BoundingBox(l=0, t=top, r=width, b=bottom, coord_origin=CoordOrigin.TOPLEFT))
            if bottom >= height:
                break
            top += step
        return rects

    def post_process_cells(self, ocr_cells: list[TextCell], page: Page, conv_res: ConversionResult, priority=None):
        cells = dedupe_overlapping(ocr_cells)
        # Docling orders a block's lines by cell index. After dedupe, neighbouring lines in an overlap
        # zone can come from different bands, so renumber top-to-bottom (layout blocks separate columns)
        cells.sort(key=lambda c: (round(c.rect.to_bounding_box().t), c.rect.to_bounding_box().l))
        for i, cell in enumerate(cells):
            cell.index = i
        super().post_process_cells(cells, page, conv_res, priority)


def dedupe_overlapping(cells: list[TextCell], min_overlap: float = 0.5) -> list[TextCell]:
    """Drop cells mostly covered by a larger one: the same line read twice where bands
    overlap, or a line clipped at a band edge (the clipped read is always the smaller box)."""
    boxes = [c.rect.to_bounding_box().to_top_left_origin(page_height=0) for c in cells]
    order = sorted(range(len(cells)), key=lambda i: boxes[i].area(), reverse=True)
    kept: list[int] = []
    for i in order:
        area = boxes[i].area()
        if area and any(boxes[i].intersection_area_with(boxes[k]) / area > min_overlap for k in kept):
            continue
        kept.append(i)
    return [cells[i] for i in sorted(kept)]


def register() -> None:
    """Add this engine to Docling's (cached) OCR factory. Call before building a converter."""
    factory = get_ocr_factory(allow_external_plugins=False)
    if BandedOcrMacOptions not in factory._classes:
        factory.register(BandedOcrMacModel, "newsrag", __name__)
