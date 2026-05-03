"""
Comprehensive PDF structure analyzer for academic papers
"""
import sys
import re
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pdfplumber"])
    import pdfplumber

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    pass

def analyze_pdf(pdf_path):
    """Analyze PDF structure comprehensively"""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING: {Path(pdf_path).name}")
    print(f"{'='*80}\n")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"TOTAL PAGES: {total_pages}\n")
        
        # Extract all text
        all_text = ""
        page_content = {}
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            all_text += text
            page_content[page_num] = text
            
            # Debug: show first page content
            if page_num == 1:
                print("="*80)
                print("FIRST PAGE CONTENT (RAW):")
                print("="*80)
                print(text[:1000])
                print("\n")
        
        # Extract metadata
        print("="*80)
        print("DOCUMENT METADATA:")
        print("="*80)
        metadata = pdf.metadata
        if metadata:
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        else:
            print("  No metadata found")
        print()
        
        # Parse structure
        print("="*80)
        print("DOCUMENT STRUCTURE ANALYSIS:")
        print("="*80)
        
        # Find title (usually first significant line)
        lines = all_text.split('\n')
        title = None
        for line in lines[:20]:
            if len(line.strip()) > 20 and len(line.strip()) < 200:
                title = line.strip()
                break
        
        if title:
            print(f"\n📄 LIKELY TITLE:\n  {title}\n")
        
        # Extract section headings
        section_pattern = r'^(Abstract|Introduction|Related Works?|Methods?|Experiments?|Results?|Discussion|Conclusion|References|Appendix|Appendices|Acknowledgments?)'
        sections = {}
        section_order = []
        
        for page_num, text in page_content.items():
            for line in text.split('\n'):
                match = re.match(section_pattern, line.strip(), re.IGNORECASE)
                if match:
                    section_name = match.group(1)
                    if section_name not in sections:
                        sections[section_name] = {'start_page': page_num, 'lines': []}
                        section_order.append(section_name)
                    sections[section_name]['lines'].append(line)
        
        print(f"\n📑 MAJOR SECTIONS FOUND ({len(sections)}):\n")
        for i, section in enumerate(section_order, 1):
            start_page = sections[section]['start_page']
            print(f"  {i}. {section:.<40} (starts p. {start_page})")
        
        # Find subsections
        subsection_pattern = r'^(1\.|2\.|3\.|4\.|5\.|[A-Z][0-9]|[A-Z]\.[0-9]|\d+\.\d+\.?)'
        print(f"\n\n🔍 SUBSECTION DETECTION:\n")
        
        subsections_by_major = {}
        for page_num, text in page_content.items():
            lines = text.split('\n')
            for line in lines:
                # Look for section markers
                if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line.strip()):
                    stripped = line.strip()
                    if stripped not in str(subsections_by_major):
                        major = None
                        for sec in section_order:
                            if sections[sec]['start_page'] <= page_num:
                                major = sec
                        if major:
                            if major not in subsections_by_major:
                                subsections_by_major[major] = []
                            subsections_by_major[major].append({
                                'text': stripped,
                                'page': page_num
                            })
        
        if subsections_by_major:
            for major_section, subs in subsections_by_major.items():
                print(f"\n  Under {major_section}:")
                for sub in subs[:5]:  # Limit to first 5
                    print(f"    - {sub['text'][:70]} (p. {sub['page']})")
                if len(subs) > 5:
                    print(f"    ... and {len(subs)-5} more")
        
        # Count figures and tables
        print(f"\n\n📊 FIGURES & TABLES:\n")
        
        fig_pattern = r'[Ff]igure\s+(\d+[a-z]?):?'
        table_pattern = r'[Tt]able\s+(\d+):?'
        
        figures = re.findall(fig_pattern, all_text)
        tables = re.findall(table_pattern, all_text)
        
        print(f"  Figures found: {len(figures)} references")
        print(f"    Numbers: {sorted(set(figures))}")
        print(f"\n  Tables found: {len(tables)} references")
        print(f"    Numbers: {sorted(set(tables))}")
        
        # Find equations/algorithms
        print(f"\n\n⚙️ SPECIAL ELEMENTS:\n")
        
        equations = all_text.count('$$') // 2
        algorithms = len(re.findall(r'[Aa]lgorithm\s+\d+', all_text))
        theorems = len(re.findall(r'[Tt]heorem\s+\d+', all_text))
        definitions = len(re.findall(r'[Dd]efinition\s+\d+', all_text))
        lemmas = len(re.findall(r'[Ll]emma\s+\d+', all_text))
        propositions = len(re.findall(r'[Pp]roposition\s+\d+', all_text))
        
        print(f"  Equations/Math blocks: {equations}")
        print(f"  Algorithms: {algorithms}")
        print(f"  Theorems: {theorems}")
        print(f"  Definitions: {definitions}")
        print(f"  Lemmas: {lemmas}")
        print(f"  Propositions: {propositions}")
        
        # Abstract analysis
        print(f"\n\n📝 ABSTRACT:\n")
        abstract_match = re.search(r'[Aa]bstract[^\n]*\n+(.*?)(?=\n\n|\n[A-Z][a-z]*\s|Introduction)', all_text, re.DOTALL)
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            abstract_lines = abstract_text.split('\n')
            abstract_clean = ' '.join([line.strip() for line in abstract_lines])[:300]
            print(f"  Length: ~{len(abstract_clean)} characters")
            print(f"  Preview: {abstract_clean}...")
        
        # Citation analysis
        print(f"\n\n📚 CITATIONS & REFERENCES:\n")
        citations = re.findall(r'\[\d+(?:,\s*\d+)*\]', all_text)
        unique_citations = len(set(re.findall(r'\[\d+\]', all_text)))
        print(f"  Citation references found: {unique_citations}")
        print(f"  Citation instances: {len(citations)}")
        
        # Find references section and count entries
        ref_match = re.search(r'[Rr]eferences.*?(?=\n[A-Z]|$)', all_text, re.DOTALL)
        if ref_match:
            ref_text = ref_match.group(0)
            ref_entries = len(re.findall(r'\[\d+\]', ref_text))
            print(f"  Reference entries: ~{ref_entries}")
        
        # Appendix info
        print(f"\n\n📎 APPENDICES:\n")
        appendix_match = re.search(r'[Aa]ppendix\s*([A-Z])?', all_text)
        if appendix_match:
            appendix_sections = re.findall(r'[Aa]ppendix\s*([A-Z])', all_text)
            print(f"  Found {len(set(appendix_sections))} appendix sections")
            print(f"  Labels: {sorted(set(appendix_sections))}")
        else:
            print("  No appendices detected")
        
        # Page distribution
        print(f"\n\n📖 PAGE DISTRIBUTION:\n")
        if section_order and total_pages > 1:
            print(f"  Total pages: {total_pages}")
            for i, section in enumerate(section_order):
                start = sections[section]['start_page']
                # Estimate end
                if i + 1 < len(section_order):
                    end = sections[section_order[i+1]]['start_page'] - 1
                else:
                    end = total_pages
                pages = end - start + 1
                pct = (pages / total_pages) * 100
                print(f"  {section:.<30} pages {start:>3}-{end:>3} ({pages:>2} pages, {pct:>5.1f}%)")
        
        # Formatting observations
        print(f"\n\n🎨 FORMATTING OBSERVATIONS:\n")
        
        # Check for columns
        max_line_length = max([len(line) for line in lines]) if lines else 0
        print(f"  Max line length: {max_line_length} characters")
        if max_line_length < 100:
            print(f"  ✓ Likely using multi-column layout")
        
        # Check for common academic formatting
        if '\\cite' in all_text or '\\ref' in all_text:
            print(f"  ✓ LaTeX formatting detected")
        
        has_emphasis = any(marker in all_text for marker in ['**', '__', '*italic*'])
        if has_emphasis:
            print(f"  ✓ Emphasis markup detected")
        
        # Summary
        print(f"\n\n{'='*80}")
        print("SUMMARY FOR REFORMATTING:")
        print(f"{'='*80}\n")
        print(f"Structure to replicate:")
        print(f"  • {len(section_order)} major sections")
        print(f"  • ~{len(subsections_by_major)} sections with subsections")
        print(f"  • {len(figures)} figures + {len(tables)} tables")
        print(f"  • {total_pages} pages total")
        print(f"  • Citation style: Numbered references [{citations[0][1:-1] if citations else '1'}]")
        print()

if __name__ == "__main__":
    pdf_path = r"c:\Users\Dr-Prashantkumar\Documents\26970_MemEIC_A_Step_Toward_Con.pdf"
    analyze_pdf(pdf_path)
