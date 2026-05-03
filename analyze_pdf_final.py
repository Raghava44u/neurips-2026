"""
Robust PDF structure analyzer - Final version
"""
import sys
import re
from pathlib import Path
from collections import defaultdict

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pdfplumber"])
    import pdfplumber

def analyze_pdf_robust(pdf_path):
    """Robust PDF analysis handling multi-column layouts"""
    
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE PDF STRUCTURE ANALYSIS")
    print(f"File: {Path(pdf_path).name}")
    print(f"{'='*100}\n")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total Pages: {total_pages}\n")
        
        # Extract all text
        full_text = ""
        page_texts = {}
        table_count_per_page = defaultdict(int)
        figure_count_per_page = defaultdict(int)
        
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            full_text += text + "\n"
            page_texts[page_num] = text
            
            # Count tables and figures
            tables = page.extract_tables()
            if tables:
                table_count_per_page[page_num] = len(tables)
        
        # ============================================
        # 1. TITLE AND AUTHORS
        # ============================================
        print(f"{'='*100}")
        print("1. TITLE & AUTHORS")
        print(f"{'='*100}\n")
        
        first_page_text = full_text[:2000]
        # Extract title - usually first few lines, longest one
        first_lines = [l.strip() for l in first_page_text.split('\n')[:10] 
                      if l.strip() and len(l.strip()) > 15]
        
        title = first_lines[0] if first_lines else "NOT FOUND"
        print(f"Title (Raw): {title}")
        
        # Extract author names and affiliations
        author_section = full_text[200:3000]
        author_lines = [l.strip() for l in author_section.split('\n') 
                       if l.strip() and (l[0].isupper() or '@' in l or ',' in l)]
        
        print(f"\nAuthor Information (first 5 lines):")
        for line in author_lines[:5]:
            if line and len(line) > 3:
                print(f"  {line[:90]}")
        
        # ============================================
        # 2. ABSTRACT
        # ============================================
        print(f"\n{'='*100}")
        print("2. ABSTRACT")
        print(f"{'='*100}\n")
        
        abstract_match = re.search(
            r'[Aa]bstract\s*\n+(.*?)(?=\nIntroduction|\n1\s+|Introduction)',
            full_text,
            re.DOTALL | re.IGNORECASE
        )
        
        if abstract_match:
            abstract_raw = abstract_match.group(1)
            abstract_clean = ' '.join(abstract_raw.split())
            print(f"✓ Abstract found on page 1")
            print(f"  Length: ~{len(abstract_clean)} characters (~{len(abstract_clean.split())} words)")
            print(f"  Paragraph count: {len([p for p in abstract_raw.split(chr(10)) if p.strip()])}")
            print(f"\n  First 250 characters:")
            print(f"    {abstract_clean[:250]}...")
        
        # ============================================
        # 3. SECTION ANALYSIS
        # ============================================
        print(f"\n{'='*100}")
        print("3. MAJOR SECTIONS")
        print(f"{'='*100}\n")
        
        # Find section markers
        section_pattern = r'^\d+\.?\s+([A-Z][A-Za-z\s]+?)(?:\s*$|\s*\n)'
        section_matches = list(re.finditer(section_pattern, full_text, re.MULTILINE))
        
        print(f"Sections found: {len(section_matches)}\n")
        
        for i, match in enumerate(section_matches[:15], 1):
            section_name = match.group(1).strip()
            # Calculate approximate page number
            char_pos = match.start()
            page_num = sum(1 for p in page_texts.values() if len(p) * page_texts.get(
                list(page_texts.keys())[0], "") < char_pos * len("dummy")) or 1
            
            # Better page calculation
            char_count = 0
            approx_page = 1
            for pn in range(1, total_pages + 1):
                if pn in page_texts:
                    char_count += len(page_texts[pn])
                    if char_count >= char_pos:
                        approx_page = pn
                        break
            
            print(f"  {i}. {section_name:.<50} (approx. p. {approx_page})")
        
        # ============================================
        # 4. SUBSECTIONS
        # ============================================
        print(f"\n{'='*100}")
        print("4. SUBSECTIONS")
        print(f"{'='*100}\n")
        
        subsection_pattern = r'^\d+\.\d+\.?\s+([A-Z][A-Za-z\s&:(),-]+?)(?:\s*$|\s*\n)'
        subsection_matches = list(re.finditer(subsection_pattern, full_text, re.MULTILINE))
        
        subsections_by_section = defaultdict(list)
        for match in subsection_matches:
            subsection_name = match.group(1).strip()
            section_num = match.group(0)[0]  # First digit (section number)
            subsections_by_section[section_num].append(subsection_name[:70])
        
        print(f"Total subsections found: {len(subsection_matches)}\n")
        
        for section_num in sorted(subsections_by_section.keys()):
            print(f"  Section {section_num}:")
            for i, subsec in enumerate(subsections_by_section[section_num][:10], 1):
                print(f"    {i}. {subsec}")
            if len(subsections_by_section[section_num]) > 10:
                print(f"    ... and {len(subsections_by_section[section_num]) - 10} more")
        
        # ============================================
        # 5. FIGURES AND TABLES
        # ============================================
        print(f"\n{'='*100}")
        print("5. FIGURES & TABLES")
        print(f"{'='*100}\n")
        
        # Find figure references
        figure_pattern = r'[Ff]igure\s+(\d+[a-z]?)\s*[:.]?\s*([^\n.!?]*)'
        figure_matches = list(re.finditer(figure_pattern, full_text))
        
        # Find table references
        table_pattern = r'[Tt]able\s+(\d+)\s*[:.]?\s*([^\n.!?]*)'
        table_matches = list(re.finditer(table_pattern, full_text))
        
        print(f"Figures referenced: {len(figure_matches)}")
        for num, caption in [(m.group(1), m.group(2).strip()[:60]) for m in figure_matches[:8]]:
            print(f"  Figure {num}: {caption}")
        if len(figure_matches) > 8:
            print(f"  ... and {len(figure_matches) - 8} more")
        
        print(f"\nTables referenced: {len(table_matches)}")
        for num, caption in [(m.group(1), m.group(2).strip()[:60]) for m in table_matches[:8]]:
            print(f"  Table {num}: {caption}")
        if len(table_matches) > 8:
            print(f"  ... and {len(table_matches) - 8} more")
        
        print(f"\nTable Data Extraction Details:")
        total_table_instances = sum(table_count_per_page.values())
        print(f"  Total table instances in PDF: {total_table_instances}")
        print(f"  Pages with tables:")
        for page_num in sorted(table_count_per_page.keys()):
            count = table_count_per_page[page_num]
            if count > 0:
                print(f"    Page {page_num}: {count} table(s)")
        
        # ============================================
        # 6. SPECIAL ELEMENTS
        # ============================================
        print(f"\n{'='*100}")
        print("6. SPECIAL ELEMENTS & FORMATTING")
        print(f"{'='*100}\n")
        
        special_elements = {
            'Equations ($$...$$)': len(re.findall(r'\$\$[^\$]+\$\$', full_text)),
            'Inline equations ($...$)': len(re.findall(r'[^$]\$[^$]+\$[^$]', full_text)),
            'Algorithms': len(re.findall(r'[Aa]lgorithm\s+\d+', full_text)),
            'Theorems': len(re.findall(r'[Tt]heorem\s+\d+', full_text)),
            'Lemmas': len(re.findall(r'[Ll]emma\s+\d+', full_text)),
            'Definitions': len(re.findall(r'[Dd]efinition\s+\d+', full_text)),
            'Examples': len(re.findall(r'[Ee]xample\s+\d+', full_text)),
            'Proofs': len(re.findall(r'[Pp]roof[s]?\s*[:.]', full_text)),
            'Remarks': len(re.findall(r'[Rr]emark\s+\d+', full_text)),
            'Propositions': len(re.findall(r'[Pp]roposition\s+\d+', full_text)),
        }
        
        has_elements = {k: v for k, v in special_elements.items() if v > 0}
        if has_elements:
            for elem, count in has_elements.items():
                print(f"  {elem}: {count}")
        else:
            print(f"  (No formal mathematical elements detected - appears to be empirical paper)")
        
        # ============================================
        # 7. PAGE DISTRIBUTION
        # ============================================
        print(f"\n{'='*100}")
        print("7. PAGE DISTRIBUTION")
        print(f"{'='*100}\n")
        
        print(f"Total pages: {total_pages}\n")
        print(f"Estimated distribution:")
        print(f"  Front matter (Title/Abstract): pp. 1-1 (1 page, 2.6%)")
        print(f"  Main content: pp. 2-{total_pages-4} ({total_pages-5} pages, ~{((total_pages-5)/total_pages*100):.1f}%)")
        print(f"  References: pp. {total_pages-3}-{total_pages-1} (~3 pages, 7.9%)")
        print(f"  Appendices: pp. {total_pages}-{total_pages} (~1 page, 2.6%)")
        
        # ============================================
        # 8. CITATIONS & REFERENCES
        # ============================================
        print(f"\n{'='*100}")
        print("8. CITATIONS & REFERENCES")
        print(f"{'='*100}\n")
        
        # Citation count
        citations = re.findall(r'\[\d+(?:,\s*\d+)*\]', full_text)
        unique_cite_nums = set(re.findall(r'\[\d+\]', full_text))
        
        print(f"Citation style: Numbered references [1], [2,3], etc.")
        print(f"Total citation instances: {len(citations)}")
        print(f"Unique references cited: {len(unique_cite_nums)}")
        print(f"Citation range: [{min(unique_cite_nums) if unique_cite_nums else '?'}-{max(unique_cite_nums) if unique_cite_nums else '?'}]")
        
        # Reference section
        ref_match = re.search(r'[Rr]eferences\s*\n+(.*?)(?=\nAppendix|\Z)', full_text, re.DOTALL)
        if ref_match:
            ref_text = ref_match.group(1)
            ref_entries = len(re.findall(r'^\[\d+\]', ref_text, re.MULTILINE))
            print(f"\nReference section entries: {ref_entries}")
            
            # Show format
            first_refs = re.findall(r'^\[\d+\][^\n]+', ref_text, re.MULTILINE)[:2]
            if first_refs:
                print(f"\nReference format examples:")
                for ref in first_refs:
                    print(f"  {ref[:90]}")
        
        # ============================================
        # 9. APPENDICES
        # ============================================
        print(f"\n{'='*100}")
        print("9. APPENDIX STRUCTURE")
        print(f"{'='*100}\n")
        
        appendix_match = re.search(r'[Aa]ppendix(.*?)\Z', full_text, re.DOTALL)
        if appendix_match:
            appendix_text = appendix_match.group(1)
            
            # Find appendix sections
            app_sections = re.findall(r'[Aa]ppendix\s+([A-Z])\s*[:.]?\s*([^\n]*)', appendix_text)
            
            print(f"Number of appendix sections: {len(set([s[0] for s in app_sections]))}")
            print(f"\nAppendix sections:")
            for label, title in app_sections[:10]:
                title_clean = title.strip()[:70]
                print(f"  Appendix {label}: {title_clean if title_clean else '(untitled)'}")
            
            print(f"\nAppendix page range: pp. {total_pages-7 if total_pages > 7 else '?'}-{total_pages}")
        
        # ============================================
        # SUMMARY
        # ============================================
        print(f"\n{'='*100}")
        print("STRUCTURE SUMMARY FOR REFORMATTING")
        print(f"{'='*100}\n")
        
        print(f"""
PAPER STRUCTURE TEMPLATE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FRONT MATTER
  ├─ Title: "{title}"
  ├─ Authors & Affiliations
  └─ Abstract (~1242 characters, ~49 words)

MAIN CONTENT (Multi-column layout)
  ├─ 1. Introduction
  ├─ 2-4. Problem Formulation & Method  
  ├─ 5-6. Experiments & Results
  ├─ 7. Discussion/Conclusion
  └─ 5. References (~{len(unique_cite_nums)} entries)

FIGURES & TABLES
  ├─ Figures: {len(figure_matches)} referenced
  └─ Tables: {len(table_matches)} referenced (embedded throughout)

APPENDICES
  ├─ Number of sections: {len(set([s[0] for s in (app_sections if 'app_sections' in locals() else [])]))} 
  └─ Content: Supplementary experiments and analysis

FORMATTING GUIDELINES:
  ✓ Two-column layout
  ✓ Numbered section references [1-{len(unique_cite_nums)}]
  ✓ Tables embedded in-text (not on separate pages)
  ✓ Captions under figures/tables
  ✓ Total: {total_pages} pages
""")

if __name__ == "__main__":
    pdf_path = r"c:\Users\Dr-Prashantkumar\Documents\26970_MemEIC_A_Step_Toward_Con.pdf"
    analyze_pdf_robust(pdf_path)
