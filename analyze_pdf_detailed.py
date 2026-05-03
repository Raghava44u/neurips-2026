"""
Enhanced PDF structure analyzer - handles multi-column layouts better
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

def extract_headers_from_page(page_text):
    """Extract headers and structure from a page"""
    lines = page_text.split('\n')
    headers = []
    
    # Common section patterns
    patterns = [
        (r'^([A-Z][A-Z\s]+)$', 'MAJOR'),  # All caps
        (r'^\d+\.\s+([A-Z][^:]+)(?::|\s|$)', 'MAJOR'),  # Numbered sections
        (r'^\d+\.\d+\s+([A-Z].+)(?::|\s|$)', 'SUBSECTION'),  # Numbered subsections
        (r'^[A-Z][a-z\s]+(?:\s[A-Z])?$', 'POSSIBLE'),  # Title case
    ]
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 3:
            continue
        if len(line_stripped) > 100:
            continue
            
        for pattern, level in patterns:
            if re.match(pattern, line_stripped):
                headers.append({
                    'text': line_stripped,
                    'level': level,
                    'length': len(line_stripped)
                })
                break
    
    return headers

def analyze_pdf_detailed(pdf_path):
    """Detailed PDF analysis"""
    
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE PDF STRUCTURE ANALYSIS")
    print(f"File: {Path(pdf_path).name}")
    print(f"{'='*100}\n")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        # Extract all text and structure
        full_text = ""
        page_texts = {}
        page_headers = defaultdict(list)
        
        print(f"Total Pages: {total_pages}")
        print(f"Extracting content...\n")
        
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            full_text += text + "\n"
            page_texts[page_num] = text
            
            # Extract tables from this page
            tables = page.extract_tables()
            if tables:
                print(f"  Page {page_num}: Found {len(tables)} table(s)")
            
            # Extract headers
            headers = extract_headers_from_page(text)
            if headers:
                page_headers[page_num] = headers
        
        # ============================================
        # 1. DOCUMENT TITLE AND AUTHORS
        # ============================================
        print(f"\n{'='*100}")
        print("1. DOCUMENT TITLE AND AUTHORS")
        print(f"{'='*100}\n")
        
        # Extract title (usually in first 500 chars on lines 1-2)
        first_page_lines = page_texts[1].split('\n')[:20]
        title_candidates = [l.strip() for l in first_page_lines 
                           if l.strip() and 20 < len(l.strip()) < 200 and l.strip()[0].isupper()]
        
        print(f"Title: {title_candidates[0] if title_candidates else 'NOT FOUND'}")
        
        # Find authors section
        author_section = full_text[500:3000]
        print(f"\nAuthors/Affiliations Found:")
        author_lines = [l.strip() for l in author_section.split('\n') if l.strip() 
                       and (l.strip()[0].isupper() or '@' in l)][:15]
        for line in author_lines[:10]:
            if line:
                print(f"  {line}")
        
        # ============================================
        # 2. ABSTRACT STRUCTURE
        # ============================================
        print(f"\n{'='*100}")
        print("2. ABSTRACT STRUCTURE")
        print(f"{'='*100}\n")
        
        abstract_match = re.search(
            r'[Aa]bstract[\s:]*\n+(.*?)(?=\n\n\d+\s|Introduction|1\s+Intro)',
            full_text,
            re.DOTALL | re.IGNORECASE
        )
        
        if abstract_match:
            abstract_text = abstract_match.group(1)
            abstract_clean = ' '.join(abstract_text.split())
            abstract_words = len(abstract_clean.split())
            print(f"Abstract found on page 1")
            print(f"  Word count: ~{abstract_words}")
            print(f"  Character count: {len(abstract_clean)}")
            print(f"  Line count: {len(abstract_match.group(1).split(chr(10)))}")
            print(f"\n  Preview (first 300 chars):")
            print(f"    {abstract_clean[:300]}...")
        
        # ============================================
        # 3. MAJOR SECTIONS IN ORDER
        # ============================================
        print(f"\n{'='*100}")
        print("3. MAJOR SECTIONS IN ORDER")
        print(f"{'='*100}\n")
        
        # Look for section headers more carefully
        section_regex = r'^\d+\.\s+([A-Z][^\n]+?)(?:\s*$|\n)'
        section_matches = [(m.start(), m.group(1), list(page_texts.keys())[
            sum(1 for p in page_texts.values() if len(full_text[:m.start()]) > len(p))
        ]) for m in re.finditer(section_regex, full_text, re.MULTILINE)]
        
        # Better approach: find section markers
        major_sections = []
        known_sections = ['Introduction', 'Related', 'Method', 'Experiment', 'Result', 
                         'Discussion', 'Conclusion', 'References', 'Appendix', 'Supplement']
        
        for i, known_sec in enumerate(known_sections):
            pattern = rf'\b{known_sec}[s]?\b'
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
            if matches:
                first_match = matches[0]
                # Find page number
                char_pos = first_match.start()
                page_num = 1
                char_count = 0
                for p in range(1, total_pages + 1):
                    if p in page_texts:
                        char_count += len(page_texts[p])
                        if char_count >= char_pos:
                            page_num = p
                            break
                
                major_sections.append({
                    'name': known_sec,
                    'page': page_num,
                    'position': char_pos
                })
        
        major_sections.sort(key=lambda x: x['position'])
        
        for i, sec in enumerate(major_sections, 1):
            print(f"  {i}. {sec['name']:.<40} Page {sec['page']:>2}")
        
        # ============================================
        # 4. SUBSECTIONS
        # ============================================
        print(f"\n{'='*100}")
        print("4. SUBSECTIONS UNDER EACH MAJOR SECTION")
        print(f"{'='*100}\n")
        
        subsection_regex = r'^\d+\.\d+\.?\s+([A-Z][^\n]*?)(?:\s*$|\n)'
        subsections_found = defaultdict(list)
        
        for match in re.finditer(subsection_regex, full_text, re.MULTILINE):
            subsection_text = match.group(1).strip()
            # Try to figure out which major section this belongs to
            section_num = match.group(0)[0]
            subsections_found[section_num].append(subsection_text[:70])
        
        if subsections_found:
            for major_num in sorted(subsections_found.keys()):
                print(f"  Section {major_num}:")
                for sub_text in subsections_found[major_num][:8]:
                    print(f"    - {sub_text}")
                if len(subsections_found[major_num]) > 8:
                    print(f"    ... and {len(subsections_found[major_num]) - 8} more")
        else:
            print("  Limited subsection structure detected (may be due to multi-column layout)")
        
        # ============================================
        # 5. FIGURES AND TABLES
        # ============================================
        print(f"\n{'='*100}")
        print("5. FIGURES & TABLES")
        print(f"{'='*100}\n")
        
        figures = re.findall(r'[Ff]igure\s+(\d+[a-z]?)\s*[:.]?\s*([^.!?]*[.!?])', full_text)
        tables = re.findall(r'[Tt]able\s+(\d+)\s*[:.]?\s*([^.!?]*[.!?])', full_text)
        
        print(f"Figures: {len(figures)}")
        for num, caption in figures[:10]:
            caption_clean = caption.strip()[:60]
            print(f"  Fig {num}: {caption_clean}")
        
        print(f"\nTables: {len(tables)}")
        for num, caption in tables[:10]:
            caption_clean = caption.strip()[:60]
            print(f"  Tab {num}: {caption_clean}")
        
        # ============================================
        # 6. SPECIAL ELEMENTS
        # ============================================
        print(f"\n{'='*100}")
        print("6. SPECIAL ELEMENTS & FORMATTING")
        print(f"{'='*100}\n")
        
        elements = {
            'Equations (inline/display)': len(re.findall(r'\$[^$]+\$|\\\[.+?\\\]', full_text, re.DOTALL)),
            'Algorithms': len(re.findall(r'[Aa]lgorithm\s+\d+', full_text)),
            'Theorems': len(re.findall(r'[Tt]heorem\s+\d+', full_text)),
            'Lemmas': len(re.findall(r'[Ll]emma\s+\d+', full_text)),
            'Definitions': len(re.findall(r'[Dd]efinition\s+\d+', full_text)),
            'Examples': len(re.findall(r'[Ee]xample\s+\d+', full_text)),
            'Proofs': len(re.findall(r'[Pp]roof[s]?:', full_text)),
            'Remarks': len(re.findall(r'[Rr]emark\s+\d+', full_text)),
        }
        
        for elem_type, count in elements.items():
            if count > 0:
                print(f"  {elem_type}: {count}")
        
        # ============================================
        # 7. PAGE DISTRIBUTION
        # ============================================
        print(f"\n{'='*100}")
        print("7. PAGE DISTRIBUTION & CONTENT DENSITY")
        print(f"{'='*100}\n")
        
        print(f"Total pages: {total_pages}\n")
        
        # Estimate distribution based on section positions
        if major_sections:
            print("Section Page Distribution:")
            for i, sec in enumerate(major_sections):
                if i + 1 < len(major_sections):
                    end_page = major_sections[i + 1]['page']
                    pages_used = end_page - sec['page']
                else:
                    pages_used = total_pages - sec['page'] + 1
                    end_page = total_pages
                
                pct = (pages_used / total_pages) * 100 if total_pages > 0 else 0
                print(f"  {sec['name']:.<30} pp. {sec['page']:>2}-{end_page:<2} ({pages_used:>2} pages, {pct:>5.1f}%)")
        
        # ============================================
        # 8. CITATIONS AND REFERENCES
        # ============================================
        print(f"\n{'='*100}")
        print("8. CITATIONS & REFERENCES")
        print(f"{'='*100}\n")
        
        citations = re.findall(r'\[\d+(?:,\s*\d+)*\]', full_text)
        unique_cites = set(re.findall(r'\[\d+\]', full_text))
        
        print(f"Citation instances: {len(citations)}")
        print(f"Unique citations: {len(unique_cites)}")
        print(f"Citation style: Numbered, e.g., [1], [2,3], etc.")
        
        # Find reference section
        ref_section = re.search(r'[Rr]eferences.*?\n(.*?)(?=\nAppendix|\n[A-Z]|\Z)', full_text, re.DOTALL)
        if ref_section:
            ref_lines = [l.strip() for l in ref_section.group(1).split('\n') 
                        if l.strip() and l.strip()[0] in '[0123456789']
            print(f"Reference entries: ~{len(ref_lines)}")
            print(f"\nReference format (examples):")
            for line in ref_lines[:3]:
                if line:
                    print(f"  {line[:80]}")
        
        # ============================================
        # 9. APPENDICES
        # ============================================
        print(f"\n{'='*100}")
        print("9. APPENDIX STRUCTURE")
        print(f"{'='*100}\n")
        
        appendix_sections = re.findall(r'[Aa]ppendix\s+([A-Z])\s*[:.]?\s*([^:\n]*)', full_text)
        
        if appendix_sections:
            print(f"Appendix sections: {len(appendix_sections)}")
            for label, title in appendix_sections:
                print(f"  Appendix {label}: {title[:60]}")
        else:
            print("No structured appendices found")
        
        # ============================================
        # SUMMARY & FORMATTING GUIDE
        # ============================================
        print(f"\n{'='*100}")
        print("REFORMATTING GUIDE SUMMARY")
        print(f"{'='*100}\n")
        
        print("""
STRUCTURE TO REPLICATE:
├── Document Type: NeurIPS/Conference Paper
├── Page Count: 38 pages
├── Layout: Multi-column (2 columns)
├── Citation System: Numbered [1], [2], etc.
│
├── Front Matter:
│   ├── Title + Authors + Affiliations (Page 1)
│   └── Abstract (Page 1)
│
├── Main Sections:
│   ├── Introduction
│   ├── Problem Formulation & Benchmark Design
│   ├── Related Works
│   ├── Method
│   ├── Experiments
│   ├── Results
│   └── Discussion/Conclusion
│
├── Figures: Multiple figures with captions
├── Tables: Multiple tables with captions
│
├── References:
│   ├── ~50 references
│   ├── Format: Numbered [1-50]
│   └── Location: After main content (before appendices)
│
└── Appendices (8 sections):
    ├── Appendix A - H
    ├── Each with subsections
    └── Varying lengths
""")

if __name__ == "__main__":
    pdf_path = r"c:\Users\Dr-Prashantkumar\Documents\26970_MemEIC_A_Step_Toward_Con.pdf"
    analyze_pdf_detailed(pdf_path)
