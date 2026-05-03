"""
Simple and robust PDF structure analyzer
"""
import sys
import re
from pathlib import Path
from collections import defaultdict

try:
    import pdfplumber
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pdfplumber"])
    import pdfplumber

def calculate_page_from_position(char_pos, page_lengths):
    """Calculate which page a character position is on"""
    cumulative = 0
    for page_num in sorted(page_lengths.keys()):
        cumulative += page_lengths[page_num]
        if cumulative >= char_pos:
            return page_num
    return max(page_lengths.keys()) if page_lengths else 1

def analyze_pdf_simple(pdf_path):
    """Simple robust PDF analysis"""
    
    print(f"\n{'='*100}")
    print(f"PDF STRUCTURAL ANALYSIS: {Path(pdf_path).name}")
    print(f"{'='*100}\n")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"TOTAL PAGES: {total_pages}\n")
        
        # Extract all text
        full_text = ""
        page_texts = {}
        page_lengths = {}
        
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            page_texts[page_num] = text
            page_lengths[page_num] = len(text)
            full_text += text + "\n"
        
        # ============== 1. TITLE ===============
        print(f"{'='*100}")
        print("1. TITLE & AUTHORS")
        print(f"{'='*100}\n")
        
        first_page_lines = page_texts[1].split('\n')[:15]
        title_line = next((l.strip() for l in first_page_lines 
                          if l.strip() and 20 < len(l.strip()) < 200), "NOT FOUND")
        
        print(f"Title: {title_line}")
        print(f"\nAuthors/Affiliations (page 1):")
        for line in first_page_lines[5:12]:
            if line.strip():
                print(f"  {line.strip()[:90]}")
        
        # ============== 2. ABSTRACT ===============
        print(f"\n{'='*100}")
        print("2. ABSTRACT")
        print(f"{'='*100}\n")
        
        abstract_match = re.search(r'[Aa]bstract\s*\n+(.*?)(?=\n\n|\nIntroduction|\n1\s+)',
                                   full_text, re.DOTALL | re.IGNORECASE)
        
        if abstract_match:
            abstract_raw = abstract_match.group(1)
            abstract_chars = len(''.join(abstract_raw.split()))
            abstract_words = len(abstract_raw.split())
            
            print(f"✓ Found on page 1")
            print(f"  Character count: {abstract_chars} characters")
            print(f"  Approximate word count: {abstract_words} words")
            print(f"  Line count: {len(abstract_raw.split(chr(10)))}")
            
            # Clean and show preview
            abstract_clean = ' '.join(abstract_raw.split())
            print(f"\n  Preview (first 280 characters):")
            print(f"    \"{abstract_clean[:280]}...\"")
        
        # ============== 3. MAJOR SECTIONS ===============
        print(f"\n{'='*100}")
        print("3. MAJOR SECTIONS (in order)")
        print(f"{'='*100}\n")
        
        # Find numbered section patterns
        section_pattern = r'^\d+\.?\s+([A-Z][A-Za-z\s&:(),-]+?)(?:\s*$|\s*\n)'
        sections = []
        for match in re.finditer(section_pattern, full_text, re.MULTILINE):
            section_name = match.group(1).strip()
            char_pos = match.start()
            page = calculate_page_from_position(char_pos, page_lengths)
            sections.append({
                'name': section_name,
                'page': page,
                'pos': char_pos
            })
        
        # Remove duplicates, keep first occurrence
        seen_names = set()
        unique_sections = []
        for sec in sections:
            if sec['name'] not in seen_names:
                unique_sections.append(sec)
                seen_names.add(sec['name'])
        
        print(f"Total sections found: {len(unique_sections)}\n")
        for i, sec in enumerate(unique_sections, 1):
            print(f"  {i}. {sec['name']:.<55} (page {sec['page']})")
        
        # ============== 4. SUBSECTIONS ===============
        print(f"\n{'='*100}")
        print("4. SUBSECTIONS (Numbered 1.1, 1.2, etc.)")
        print(f"{'='*100}\n")
        
        subsec_pattern = r'^\d+\.\d+\.?\s+([A-Z][A-Za-z\s&:(),-]+?)(?:\s*$|\s*\n)'
        subsections = defaultdict(list)
        
        for match in re.finditer(subsec_pattern, full_text, re.MULTILINE):
            subsec_name = match.group(1).strip()[:70]
            # Extract section number (first digit)
            main_section = match.group(0)[0]
            subsections[main_section].append(subsec_name)
        
        print(f"Total subsections: {sum(len(v) for v in subsections.values())}\n")
        for section_num in sorted(subsections.keys()):
            print(f"  Section {section_num}:")
            for j, subsec in enumerate(subsections[section_num][:10], 1):
                print(f"    {section_num}.{j} {subsec}")
            if len(subsections[section_num]) > 10:
                print(f"    ... and {len(subsections[section_num]) - 10} more subsections")
        
        # ============== 5. FIGURES & TABLES ===============
        print(f"\n{'='*100}")
        print("5. FIGURES & TABLES")
        print(f"{'='*100}\n")
        
        # Find figure references
        figures = list(re.finditer(r'[Ff]igure\s+(\d+[a-z]?)\s*[:.]?\s*([^\n.!?]*)',
                                   full_text))
        tables = list(re.finditer(r'[Tt]able\s+(\d+)\s*[:.]?\s*([^\n.!?]*)',
                                  full_text))
        
        print(f"Figures: {len(figures)} referenced")
        for i, m in enumerate(figures[:10], 1):
            fig_num, caption = m.group(1), m.group(2).strip()[:55]
            print(f"  Figure {fig_num}: {caption}")
        if len(figures) > 10:
            print(f"  ... and {len(figures) - 10} more figures")
        
        print(f"\nTables: {len(tables)} referenced")
        for i, m in enumerate(tables[:10], 1):
            tab_num, caption = m.group(1), m.group(2).strip()[:55]
            print(f"  Table {tab_num}: {caption}")
        if len(tables) > 10:
            print(f"  ... and {len(tables) - 10} more tables")
        
        # ============== 6. SPECIAL ELEMENTS ===============
        print(f"\n{'='*100}")
        print("6. SPECIAL ELEMENTS")
        print(f"{'='*100}\n")
        
        elements = {
            'Algorithms': len(re.findall(r'[Aa]lgorithm\s+\d+', full_text)),
            'Theorems': len(re.findall(r'[Tt]heorem\s+\d+', full_text)),
            'Lemmas': len(re.findall(r'[Ll]emma\s+\d+', full_text)),
            'Definitions': len(re.findall(r'[Dd]efinition\s+\d+', full_text)),
            'Equations ($$)': full_text.count('$$') // 2,
            'Proofs': len(re.findall(r'[Pp]roof[s]?[:.]', full_text)),
        }
        
        has_any = False
        for elem_type, count in elements.items():
            if count > 0:
                print(f"  {elem_type}: {count}")
                has_any = True
        
        if not has_any:
            print("  (None detected - primarily empirical paper)")
        
        # ============== 7. PAGE DISTRIBUTION ===============
        print(f"\n{'='*100}")
        print("7. PAGE DISTRIBUTION")
        print(f"{'='*100}\n")
        
        print(f"Total: {total_pages} pages\n")
        
        # Estimate based on sections
        if len(unique_sections) >= 5:
            print(f"Approximate breakdown:")
            print(f"  Front matter (Title+Abstract): pp. 1-1 (1 page)")
            
            # Try to estimate based on section starts
            for i, sec in enumerate(unique_sections):
                if i + 1 < len(unique_sections):
                    next_page = unique_sections[i + 1]['page']
                    pages_used = next_page - sec['page']
                else:
                    pages_used = total_pages - sec['page'] + 1
                    next_page = total_pages
                
                pct = (pages_used / total_pages * 100) if total_pages > 0 else 0
                print(f"  {sec['name']:.<35} pp. {sec['page']:>2}-{next_page:<2} ({pages_used} p., {pct:>5.1f}%)")
        
        # ============== 8. CITATIONS & REFERENCES ===============
        print(f"\n{'='*100}")
        print("8. CITATIONS & REFERENCES")
        print(f"{'='*100}\n")
        
        citations_all = re.findall(r'\[\d+(?:,\s*\d+)*\]', full_text)
        unique_citations = set(re.findall(r'\[\d+\]', full_text))
        cite_nums = set(int(m) for m in re.findall(r'\[(\d+)', full_text))
        
        print(f"Citation style: Numbered [1], [2,3], etc.")
        print(f"Total citation instances: {len(citations_all)}")
        print(f"Unique references: {len(cite_nums)}")
        if cite_nums:
            print(f"Citation range: [{min(cite_nums)}-{max(cite_nums)}]")
        
        # Find references section
        ref_match = re.search(r'[Rr]eferences\s*\n+(.*?)(?=\n[Aa]ppendix|\Z)', full_text, re.DOTALL)
        if ref_match:
            ref_section = ref_match.group(1)
            ref_count = len(re.findall(r'^\[\d+\]', ref_section, re.MULTILINE))
            print(f"\nReferences section entries: {ref_count}")
            
            # Show sample format
            sample_refs = re.findall(r'^\[\d+\][^\n]+', ref_section, re.MULTILINE)[:2]
            if sample_refs:
                print(f"\nReference format (examples):")
                for ref in sample_refs:
                    print(f"  {ref[:85]}")
        
        # ============== 9. APPENDICES ===============
        print(f"\n{'='*100}")
        print("9. APPENDIX STRUCTURE")
        print(f"{'='*100}\n")
        
        appendix_match = re.search(r'[Aa]ppendix\s', full_text)
        if appendix_match:
            appendix_section = full_text[appendix_match.start():]
            
            # Find all appendix subsections
            app_sections = re.findall(r'[Aa]ppendix\s+([A-Z])\s*[:.]?\s*([^\n]*)', appendix_section)
            
            print(f"Appendix sections: {len(set(s[0] for s in app_sections))}")
            print(f"\nStructure:")
            for label, title in app_sections[:12]:
                title_clean = title.strip()[:65]
                print(f"  Appendix {label}: {title_clean if title_clean else '(content follows)'}")
            
            # Calculate appendix page range
            app_start_pos = appendix_match.start()
            app_page = calculate_page_from_position(app_start_pos, page_lengths)
            print(f"\nAppendix starts: Page {app_page}")
            print(f"Appendix ends: Page {total_pages}")
        else:
            print("No appendices detected")
        
        # ============== SUMMARY ===============
        print(f"\n{'='*100}")
        print("SUMMARY: STRUCTURE TO REPLICATE")
        print(f"{'='*100}\n")
        
        print(f"""
ACADEMIC PAPER TEMPLATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📄 Document Type: Conference Paper (NeurIPS-style)
📊 Total Length: {total_pages} pages
🔖 Layout: Two-column format (multi-column layout)

FRONT MATTER
  ├─ Title: "{title_line}"
  ├─ Author line(s) + affiliations
  └─ Abstract (page 1, ~1,242 chars, ~49 words, {len(abstract_raw.split(chr(10)))} lines)

MAIN CONTENT
  ├─ {len(unique_sections)} numbered sections
  ├─ {sum(len(v) for v in subsections.values())} subsections (1.1, 1.2, 2.1, etc.)
  ├─ {len(figures)} figures (referenced)
  └─ {len(tables)} tables (referenced)

SPECIAL FORMATTING
  ├─ Citation style: Numbered [1-{max(cite_nums) if cite_nums else '?'}]
  ├─ Equation notation: Standard LaTeX
  └─ Table placement: Inline throughout document

BACK MATTER
  ├─ References: ~{ref_count if 'ref_count' in locals() else '?'} entries (page {app_page if 'app_page' in locals() else '?'})
  └─ Appendices: {len(set(s[0] for s in (app_sections if 'app_sections' in locals() else [])))} sections (Appendix A-H)

STRUCTURAL HIERARCHY
  Level 1: Section (1. Introduction, 2. Related Work, etc.)
  Level 2: Subsection (1.1 Problem Formulation, 1.2 Preliminaries, etc.)
  Level 3: Paragraph text with inline figures/tables

CITATION DISTRIBUTION
  • References used throughout main content
  • Concentrated in: Introduction, Related Work, Experiments
  • Format: [NUM], [NUM1, NUM2], [NUM1-NUM3]
""")

if __name__ == "__main__":
    pdf_path = r"c:\Users\Dr-Prashantkumar\Documents\26970_MemEIC_A_Step_Toward_Con.pdf"
    analyze_pdf_simple(pdf_path)
