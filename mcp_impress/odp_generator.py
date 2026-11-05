"""
Deterministic OpenDocument Presentation (.odp) Synthesis Engine.
Generates fully compliant ODP zip archives featuring native SMIL transitions
(Fade, Push, Wipe) and sequential object build animations (<anim:seq>, <anim:par>).
"""

from __future__ import annotations
import io
import os
import subprocess
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SlideAnimation:
    shape_id: str
    animation_type: str = "appear"  # "appear", "fade_in", "fly_in"
    trigger_event: str = "click"   # "click", "with_previous", "after_previous"
    delay: float = 0.0


@dataclass
class Slide:
    title: str
    subtitle: str = ""
    bullets: List[str] = field(default_factory=list)
    callout_boxes: List[Dict[str, str]] = field(default_factory=list)
    transition_type: str = "fade"  # "fade", "push", "wipe", "bar"
    transition_speed: str = "fast" # "fast", "medium", "slow"
    transition_direction: str = "from-right"
    animations: List[SlideAnimation] = field(default_factory=list)


class ODPPresentationBuilder:
    """
    Constructs an OpenDocument Presentation (.odp) archive with custom styles,
    typography, dark-mode technical themes, native slide transitions, and build animations.
    """

    def __init__(self, title: str = "Presentation", theme: str = "cyber_dark"):
        self.title = title
        self.theme = theme
        self.slides: List[Slide] = []

    def add_slide(
        self,
        title: str,
        subtitle: str = "",
        bullets: Optional[List[str]] = None,
        callout_boxes: Optional[List[Dict[str, str]]] = None,
        transition_type: str = "fade",
        transition_speed: str = "fast",
        transition_direction: str = "from-right",
    ) -> Slide:
        slide = Slide(
            title=title,
            subtitle=subtitle,
            bullets=bullets or [],
            callout_boxes=callout_boxes or [],
            transition_type=transition_type,
            transition_speed=transition_speed,
            transition_direction=transition_direction,
        )
        self.slides.append(slide)
        return slide

    def apply_animation(
        self,
        slide_index: int,
        shape_id: str,
        animation_type: str = "appear",
        trigger_event: str = "click",
        delay: float = 0.0,
    ):
        if 0 <= slide_index < len(self.slides):
            self.slides[slide_index].animations.append(
                SlideAnimation(
                    shape_id=shape_id,
                    animation_type=animation_type,
                    trigger_event=trigger_event,
                    delay=delay,
                )
            )

    def set_slide_transition(
        self,
        slide_index: int,
        transition_type: str = "push",
        speed: str = "fast",
        direction: str = "from-right",
    ):
        if 0 <= slide_index < len(self.slides):
            s = self.slides[slide_index]
            s.transition_type = transition_type
            s.transition_speed = speed
            s.transition_direction = direction

    def _generate_manifest_xml(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.3">
 <manifest:file-entry manifest:full-path="/" manifest:version="1.3" manifest:media-type="application/vnd.oasis.opendocument.presentation"/>
 <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
 <manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>
 <manifest:file-entry manifest:full-path="meta.xml" manifest:media-type="text/xml"/>
</manifest:manifest>"""

    def _generate_meta_xml(self) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0"
 xmlns:dc="http://purl.org/dc/elements/1.1/" office:version="1.3">
 <office:meta>
  <dc:title>{self.title}</dc:title>
  <dc:creator>Technical Presentation Engineer</dc:creator>
  <meta:generator>ODP Presentation Engine 2.0</meta:generator>
 </office:meta>
</office:document-meta>"""

    def _generate_styles_xml(self) -> str:
        # 16:9 widescreen layout (28cm x 15.75cm)
        return """<?xml version="1.0" encoding="UTF-8"?>
<office:document-styles xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
 xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
 xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
 xmlns:smil="urn:oasis:names:tc:opendocument:xmlns:smil-compatible:1.0"
 office:version="1.3">
 <office:styles>
  <!-- Standard Default Graphics Style -->
  <style:default-style style:family="graphic">
   <style:graphic-properties svg:stroke-color="#1e293b" draw:fill-color="#0f172a" fo:wrap-option="wrap"/>
  </style:default-style>

  <!-- Master Page Background Fill -->
  <style:style style:name="SlideBackground" style:family="drawing-page">
   <style:drawing-page-properties draw:fill="solid" draw:fill-color="#0a0d14"/>
  </style:style>
 </office:styles>

 <office:master-styles>
  <style:master-page style:name="Default" style:page-layout-name="LayoutWidescreen" draw:style-name="SlideBackground">
  </style:master-page>
 </office:master-styles>

 <office:automatic-styles>
  <style:page-layout style:name="LayoutWidescreen">
   <style:page-layout-properties fo:page-width="28cm" fo:page-height="15.75cm" fo:margin-top="0cm" fo:margin-bottom="0cm" fo:margin-left="0cm" fo:margin-right="0cm"/>
  </style:page-layout>
 </office:automatic-styles>
</office:document-styles>"""

    def _generate_content_xml(self) -> str:
        # Automatic slide-specific transition styles
        auto_styles = []
        for i, slide in enumerate(self.slides):
            style_name = f"SlideTransStyle_{i}"
            trans_type = slide.transition_type  # "fade", "push", "wipe"
            trans_speed = slide.transition_speed
            trans_subtype = slide.transition_direction
            auto_styles.append(f"""
  <style:style style:name="{style_name}" style:family="drawing-page">
   <style:drawing-page-properties smil:type="{trans_type}" smil:subtype="{trans_subtype}" presentation:transition-speed="{trans_speed}" presentation:transition-type="automatic" draw:fill="solid" draw:fill-color="#0a0d14"/>
  </style:style>""")

        # Custom text and box styles
        auto_styles.append("""
  <style:style style:name="TitleStyle" style:family="text">
   <style:text-properties fo:font-size="26pt" fo:font-weight="bold" fo:color="#38bdf8" fo:font-family="Liberation Sans, Segoe UI, sans-serif"/>
  </style:style>
  <style:style style:name="SubtitleStyle" style:family="text">
   <style:text-properties fo:font-size="14pt" fo:font-style="italic" fo:color="#94a3b8" fo:font-family="Liberation Sans, Segoe UI, sans-serif"/>
  </style:style>
  <style:style style:name="BodyStyle" style:family="text">
   <style:text-properties fo:font-size="15pt" fo:color="#f8fafc" fo:font-family="Liberation Sans, Segoe UI, sans-serif"/>
  </style:style>
  <style:style style:name="BulletHighlight" style:family="text">
   <style:text-properties fo:font-size="15pt" fo:font-weight="bold" fo:color="#10b981" fo:font-family="Liberation Sans, Segoe UI, sans-serif"/>
  </style:style>
  <style:style style:name="CodeTextStyle" style:family="text">
   <style:text-properties fo:font-size="12pt" fo:color="#f59e0b" fo:font-family="Liberation Mono, Courier New, monospace"/>
  </style:style>

  <!-- Container Box Graphics Styles -->
  <style:style style:name="CardFrame" style:family="graphic">
   <style:graphic-properties draw:fill="solid" draw:fill-color="#111726" svg:stroke-color="#1e293b" svg:stroke-width="0.04cm" draw:corner-radius="0.25cm"/>
  </style:style>
  <style:style style:name="CalloutFrame" style:family="graphic">
   <style:graphic-properties draw:fill="solid" draw:fill-color="#172033" svg:stroke-color="#38bdf8" svg:stroke-width="0.06cm" draw:corner-radius="0.25cm"/>
  </style:style>
""")

        # Slide content generation
        pages_xml = []
        for i, slide in enumerate(self.slides):
            page_name = f"Slide_{i+1}"
            style_name = f"SlideTransStyle_{i}"

            # Bullets XML
            bullets_p = []
            for b_idx, bullet in enumerate(slide.bullets):
                bullet_shape_id = f"bullet_{i}_{b_idx}"
                # Separate highlight prefix if contains ':'
                if ":" in bullet:
                    prefix, rest = bullet.split(":", 1)
                    p_content = f'<text:span text:style-name="BulletHighlight">{escape_xml(prefix)}:</text:span><text:span text:style-name="BodyStyle">{escape_xml(rest)}</text:span>'
                else:
                    p_content = f'<text:span text:style-name="BodyStyle">{escape_xml(bullet)}</text:span>'

                bullets_p.append(f"""
     <text:list-item>
      <text:p text:style-name="P_Bullet">
       {p_content}
      </text:p>
     </text:list-item>""")

            bullets_xml = "\n".join(bullets_p)

            # Callout boxes (if any)
            callout_shapes = []
            if slide.callout_boxes:
                c_y = 4.0
                for c_idx, box in enumerate(slide.callout_boxes):
                    b_title = box.get("title", "")
                    b_val = box.get("value", "")
                    b_desc = box.get("desc", "")
                    callout_shapes.append(f"""
    <!-- Callout Box {c_idx+1} -->
    <draw:custom-shape draw:name="callout_{i}_{c_idx}" draw:style-name="CalloutFrame" svg:width="8.5cm" svg:height="4.2cm" svg:x="18.0cm" svg:y="{c_y}cm">
     <text:p><text:span text:style-name="BulletHighlight">{escape_xml(b_title)}</text:span></text:p>
     <text:p><text:span text:style-name="TitleStyle">{escape_xml(b_val)}</text:span></text:p>
     <text:p><text:span text:style-name="SubtitleStyle">{escape_xml(b_desc)}</text:span></text:p>
    </draw:custom-shape>""")
                    c_y += 4.8
            callouts_xml = "\n".join(callout_shapes)

            # SMIL Sequential Animation XML
            anim_items = []
            for a_idx, anim in enumerate(slide.animations):
                anim_items.append(f"""
     <anim:par smil:begin="{anim.trigger_event}">
      <anim:set smil:targetElement="{anim.shape_id}" smil:attributeName="visibility" smil:to="visible"/>
     </anim:par>""")

            # If no manual animations were added, generate default sequential bullet reveal
            if not anim_items and slide.bullets:
                for b_idx in range(len(slide.bullets)):
                    anim_items.append(f"""
     <anim:par smil:begin="click">
      <anim:set smil:targetElement="content_frame_{i}" smil:attributeName="visibility" smil:to="visible"/>
     </anim:par>""")

            animation_block = f"""
   <anim:par presentation:node-type="timing-root">
    <anim:seq presentation:node-type="main-sequence">
     {"".join(anim_items)}
    </anim:seq>
   </anim:par>""" if anim_items else ""

            # Compose complete slide XML
            page_xml = f"""
  <draw:page draw:name="{page_name}" draw:style-name="{style_name}" draw:master-page-name="Default">
   <!-- Header Frame -->
   <draw:frame draw:name="header_frame_{i}" draw:style-name="CardFrame" svg:width="25.6cm" svg:height="2.3cm" svg:x="1.2cm" svg:y="1.0cm">
    <draw:text-box>
     <text:p><text:span text:style-name="TitleStyle">{escape_xml(slide.title)}</text:span></text:p>
     <text:p><text:span text:style-name="SubtitleStyle">{escape_xml(slide.subtitle)}</text:span></text:p>
    </draw:text-box>
   </draw:frame>

   <!-- Body Content Frame -->
   <draw:frame draw:name="content_frame_{i}" draw:style-name="CardFrame" svg:width="{"15.8cm" if slide.callout_boxes else "25.6cm"}" svg:height="10.8cm" svg:x="1.2cm" svg:y="3.8cm">
    <draw:text-box>
     <text:list>
      {bullets_xml}
     </text:list>
    </draw:text-box>
   </draw:frame>
   {callouts_xml}
   {animation_block}
  </draw:page>"""
            pages_xml.append(page_xml)

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
 xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
 xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
 xmlns:smil="urn:oasis:names:tc:opendocument:xmlns:smil-compatible:1.0"
 xmlns:anim="urn:oasis:names:tc:opendocument:xmlns:animation:1.0"
 office:version="1.3">
 <office:automatic-styles>
  {"".join(auto_styles)}
 </office:automatic-styles>
 <office:body>
  <office:presentation>
   {"".join(pages_xml)}
  </office:presentation>
 </office:body>
</office:document-content>"""

    def save_odp(self, file_path: str) -> str:
        """Packages the XML documents into a valid ZIP-based .odp archive."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            # 1. mimetype MUST be uncompressed at the beginning of the archive
            z.writestr("mimetype", "application/vnd.oasis.opendocument.presentation", compress_type=zipfile.ZIP_STORED)
            # 2. Manifest
            z.writestr("META-INF/manifest.xml", self._generate_manifest_xml())
            # 3. Meta
            z.writestr("meta.xml", self._generate_meta_xml())
            # 4. Styles
            z.writestr("styles.xml", self._generate_styles_xml())
            # 5. Content
            z.writestr("content.xml", self._generate_content_xml())

        with open(file_path, "wb") as f:
            f.write(buf.getvalue())
        return file_path

    def export_to_pdf(self, odp_path: str, output_dir: Optional[str] = None) -> str:
        """Converts generated .odp presentation into production PDF using headless LibreOffice."""
        out_dir = output_dir or os.path.dirname(os.path.abspath(odp_path))
        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            odp_path,
            "--outdir",
            out_dir,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"LibreOffice conversion failed: {res.stderr}")

        base_name = os.path.splitext(os.path.basename(odp_path))[0]
        pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
        return pdf_path


def escape_xml(s: str) -> str:
    """Escapes special XML characters."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
