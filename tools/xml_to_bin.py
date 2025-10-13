
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def _write_records_from_iterparse(src: Path, dst: Path, record_tags: list[str]):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as out:
        context = ET.iterparse(str(src), events=("end",))
        for _, elem in context:
            tag = elem.tag or ""
            if any(tag.endswith(rt) for rt in record_tags):
                # try several common paths for the main heading
                hn = (elem.find(".//DescriptorName/String")
                      or elem.find(".//SupplementalRecordName/String")
                      or elem.find(".//SupplementaryConcept/Name/String")
                      or elem.find(".//RecordName/String")
                      or elem.find(".//Concept/PreferredConcept/Term/String")
                      or elem.find(".//Concept/TermList/Term/String"))
                heading = (hn.text or "").strip() if hn is not None and hn.text else ""
                if heading:
                    out.write("*NEWRECORD\n")
                    out.write(f"MH = {heading}\n")
                    # entries: common Term paths
                    term_nodes = list(elem.findall(".//Concept/TermList/Term/String")) + list(elem.findall(".//TermList/Term/String")) + list(elem.findall(".//SupplementalRecord/TermList/Term/String"))
                    for term in term_nodes:
                        txt = (term.text or "").strip()
                        if txt:
                            out.write(f"ENTRY = {txt}\n")
                    # tree numbers (if any)
                    for tn in elem.findall(".//TreeNumberList/TreeNumber"):
                        txt = (tn.text or "").strip()
                        if txt:
                            out.write(f"MN = {txt}\n")
                elem.clear()

def convert_descriptor_xml_to_bin(src_xml: str, dst_bin: str):
    src = Path(src_xml)
    dst = Path(dst_bin)
    if not src.exists():
        raise SystemExit(f"source not found: {src}")
    # Descriptor records use DescriptorRecord
    _write_records_from_iterparse(src, dst, ["DescriptorRecord"])

def convert_supplementary_xml_to_bin(src_xml: str, dst_bin: str):
    src = Path(src_xml)
    dst = Path(dst_bin)
    if not src.exists():
        raise SystemExit(f"source not found: {src}")
    # Supplementary files may use several record element names; include the common ones
    _write_records_from_iterparse(src, dst, ["SupplementaryRecord", "SupplementalRecord", "SupplementaryConceptRecord", "SupplementaryConcept"])

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python tools\\xml_to_bin.py descriptor <desc2020.xml> <d2020.bin>")
        print("  python tools\\xml_to_bin.py supplementary <supp2020.xml> <supp2020.bin>")
        sys.exit(2)
    mode = sys.argv[1].lower()
    if mode == "descriptor":
        convert_descriptor_xml_to_bin(sys.argv[2], sys.argv[3])
    elif mode in ("supplementary", "supp"):
        convert_supplementary_xml_to_bin(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("Unknown mode. Use 'descriptor' or 'supplementary'.")
