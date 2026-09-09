import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


OUT_PATH = Path(
    r"C:\llm_project\Improved LSTM+GAIN_pytorch\Improved LSTM+GAIN\experiment_2014_4users\guided_gain_weight_summary.docx"
)


def build_docx(paragraphs):
    body = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
        for text in paragraphs
    )

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:w10="urn:schemas-microsoft-com:office:word" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" mc:Ignorable="w14 wp14">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
      <w:cols w:space="708"/>
      <w:docGrid w:linePitch="360"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Guided GAIN Weight Summary</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
</cp:coreProperties>"""

    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>"""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)


def main():
    paragraphs = [
        "\u5fae\u8c03 GAIN \u6743\u91cd\u8bbe\u7f6e\u7b80\u8ff0",
        "\u751f\u6210\u5668\u603b\u635f\u5931\u4e3a\uff1aL_G = L_adv + alpha \u00d7 L_obs + beta \u00d7 L_P\u3002",
        "\u5176\u4e2d\uff0cL_adv \u7684\u6743\u91cd\u4e3a 1\uff1bL_obs \u7684\u6743\u91cd alpha \u5f53\u524d\u8bbe\u4e3a 100\uff1bL_P \u4e3a\u51bb\u7ed3 LSTM \u7ed9\u51fa\u7684\u9884\u6d4b\u8bef\u5dee\u635f\u5931\u3002",
        "L_P \u7684\u6743\u91cd beta \u4e0d\u662f\u4ece\u4e00\u5f00\u59cb\u5c31\u56fa\u5b9a\u4e3a 0.001\uff0c\u800c\u662f\u5728\u524d 500 \u6b21\u8fed\u4ee3\u4e2d\u4ece 0 \u7ebf\u6027\u5347\u9ad8\u5230 0.001\uff0c\u4e4b\u540e\u4fdd\u6301 0.001\u3002",
        "\u56e0\u6b64\uff0c\u5f53\u524d\u5fae\u8c03\u7b56\u7565\u7684\u6838\u5fc3\u662f\uff1a\u4ee5\u539f\u59cb GAIN \u8865\u5168\u76ee\u6807\u4e3a\u4e3b\uff0c\u7528\u8f83\u5c0f\u6743\u91cd\u7684\u9884\u6d4b\u635f\u5931\u5bf9\u751f\u6210\u5668\u8fdb\u884c\u5f15\u5bfc\u3002",
    ]
    build_docx(paragraphs)
    print(str(OUT_PATH))


if __name__ == "__main__":
    main()
