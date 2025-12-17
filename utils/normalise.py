""" Load and flatten the data from the input file before processing. Outputs pandas dataframe."""

from pathlib import Path
import pandas as pd
from typing import Optional
import re

def parse_custom_array(text: str, column_name: str = "unknown") -> list[str]:
    """Parse custom array format into a list of strings.
    
    Args:
        text: The text to parse (can be pandas Series value, string, etc.)
        column_name: Name of the column being parsed (for error messages)
    
    Returns:
        List of parsed strings
        
    Raises:
        ValueError: If the text format is invalid and cannot be parsed
    """
    if pd.isna(text) or str(text).strip() == "{}":
        return []

    text_str = str(text).strip()
    original_text = text_str
    
    if text_str.startswith('{') and text_str.endswith('}'):
        text_str = text_str[1:-1]
    
    pattern = r'(?:"([^"]*)"|([^\s,]+))'
    matches = re.findall(pattern, text_str)

    results = [quoted.strip() if quoted else unquoted.strip() for quoted, unquoted in matches]
    
    # sanity check: if no results and original text was not empty, raise error
    if original_text and original_text != "{}" and not results:
        raise ValueError(
            f"Failed to parse array format in column '{column_name}': '{original_text}'. "
            f"Expected format: {{value1, value2, ...}} or {{'value1', 'value2', ...}}"
        )
    
    return results

def get_safe(lst: list[str], index: int, default="") -> str:
    """Safely retrieve items from lists;handling length and type errors."""
    if index < len(lst):
        val = lst[index]
        if str(val).lower() in ["nan", "null", "none", "", "0"] or val is None:
            return default
        return str(val)
    return default

def clean_numeric_id(value) -> str:
    """Convert numeric ID to clean string without decimals (1.0 -> '1')."""
    if pd.isna(value):
        return ""
    # Convert to int 
    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value)

def format_penalty(has_fine, min_fine, max_fine, has_imprisonment, min_imprisonment, max_imprisonment, 
                  context: str = "") -> str:
    """Creates a formatted string for the penalties.
    
    Args:
        has_fine: Boolean or string indicating if fine applies
        min_fine: Minimum fine value
        max_fine: Maximum fine value
        has_imprisonment: Boolean or string indicating if imprisonment applies
        min_imprisonment: Minimum imprisonment value
        max_imprisonment: Maximum imprisonment value
        context: Optional context string for error messages (e.g., "section_num=123")
    
    Returns:
        Formatted penalty string
    """
    penalties = []

    try:
        is_fine_valid = str(has_fine).lower() in ['true', '1'] or (pd.notna(min_fine) and str(min_fine) != 'nan') or (pd.notna(max_fine) and str(max_fine) != 'nan')
        is_imprisonment_valid = str(has_imprisonment).lower() in ['true', '1'] or (pd.notna(min_imprisonment) and str(min_imprisonment) != 'nan') or (pd.notna(max_imprisonment) and str(max_imprisonment) != 'nan')
    except Exception as e:
        context_msg = f" ({context})" if context else ""
        raise ValueError(f"Error processing penalty values{context_msg}: {e}")

    if is_fine_valid:
        s = "Fine"
        details = []

        if pd.notna(min_fine) and str(min_fine) not in ['0', 'nan']: details.append(f"Min: {min_fine}")
        if pd.notna(max_fine) and str(max_fine) not in ['0', 'nan']: details.append(f"Max: {max_fine}")
        if details:
            s += f" ({', '.join(details)})"
        penalties.append(s)

    if is_imprisonment_valid:
        s = "Imprisonment"
        details = []
        if pd.notna(min_imprisonment) and str(min_imprisonment) not in ['0', 'nan']: details.append(f"Min: {min_imprisonment}")
        if pd.notna(max_imprisonment) and str(max_imprisonment) not in ['0', 'nan']: details.append(f"Max: {max_imprisonment}")
        if details:
            s += f" ({', '.join(details)})"
        penalties.append(s)
    return ", ".join(penalties) if penalties else "Not specified"


def load_and_flatten_data(input_file: Path, output_file: Optional[Path] = None) -> pd.DataFrame:
    """Load and flatten the data from the input file before processing. Outputs pandas DataFrame."""
    print(f"Loading {input_file}...")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    try:
        df = pd.read_excel(input_file, sheet_name=1)
    except Exception as e:
        print(f"Error loading Sheet 2 from {input_file}: {e}")
        raise

    # Validate required columns exist
    required_columns = [
        "chap_id", "section_num", "parent_section_id", "level", "title", "description", "tags",
        "has_fine", "min_fine", "max_fine", "has_imprisonment",
        "min_imprisonment", "max_imprisonment", "entity_involved",
        "action_type", "intent", "offense_category", "law_references"
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in Excel file Sheet 2: {', '.join(missing_columns)}\n"
            f"Found columns: {', '.join(df.columns.tolist())}"
        )

    # Build parent lookup map from level==0 rows
    parent_map = {}
    parents = df[df['level'] == 0]
    for _, parent_row in parents.iterrows():
        clean_section = clean_numeric_id(parent_row['section_num'])
        parent_map[clean_section] = str(parent_row['title']).strip()

    processed_rows = []
    rows_processed = 0
    rows_failed = 0

    for index, row in df.iterrows():
        try:
            # Extract row data.
            chap_id = str(row["chap_id"]) if pd.notna(row["chap_id"]) else ""
            section_num = row["section_num"]
            level = row.get("level", 0)
            parent_section_id = row.get("parent_section_id")
            title = str(row["title"]).strip()
            description = str(row["description"]).strip()
            
            tags = str(row["tags"]) if pd.notna(row["tags"]) else ""
            has_fine = row["has_fine"]
            min_fine = row["min_fine"]
            max_fine = row["max_fine"]
            has_imprisonment = row["has_imprisonment"]
            min_imprisonment = row["min_imprisonment"]
            max_imprisonment = row["max_imprisonment"]
            entity_involved = str(row["entity_involved"]) if pd.notna(row["entity_involved"]) else ""
            action_type = str(row["action_type"]) if pd.notna(row["action_type"]) else ""
            intent = str(row["intent"]) if pd.notna(row["intent"]) else ""
            offense_category = str(row["offense_category"]) if pd.notna(row["offense_category"]) else "General"
            law_references = str(row["law_references"]) if pd.notna(row["law_references"]) else ""

            # Penalty Formatting
            penalty_context = f"row_index={index}, section_num={section_num}, chap_id={chap_id}"
            penalty_text = format_penalty(
                has_fine, min_fine, max_fine,
                has_imprisonment, min_imprisonment, max_imprisonment,
                context=penalty_context
            )
            
            # Determine unique ID and context header based on level
            is_subsection = level > 0
            
            # Clean numeric IDs (strip .0 from floats like 1.0 -> 1)
            clean_section = clean_numeric_id(section_num)
            clean_parent = clean_numeric_id(parent_section_id) if pd.notna(parent_section_id) else None
            
            if is_subsection:
                # Child row: use parent.section format for ID
                unique_id = f"{clean_parent}.{clean_section}"
                parent_title = parent_map.get(clean_parent, "Unknown Parent")
                context_header = f"Parent Section: {parent_title} (Section {clean_parent})\n"
                section_label = f"Sub-section {clean_section}"
            else:
                # Parent row: use section_num as ID
                unique_id = clean_section
                context_header = ""
                section_label = f"Section {clean_section}"
            
            # Construct embedding text
            embed_text = (
                f"{context_header}"
                f"Chapter: {chap_id}\n"
                f"{section_label}: {title}\n"
                f"Content: {description}\n"
                f"Category: {offense_category}\n"
                f"Intent: {intent}\n"
                f"Entity Involved: {entity_involved}\n"
                f"Action Type: {action_type}\n"
                f"Penalties: {penalty_text}\n"
                f"Keywords: {tags}\n"
                f"References: {law_references}"
            )
            
            processed_rows.append({
                'id': unique_id,
                'text': embed_text,
                'section_num': clean_section,
                'title': title,
                'level': level,
                'parent_section_id': clean_parent,
                'chapter': chap_id,
                'is_subsection': is_subsection
            })
            
            rows_processed += 1
        except Exception as e:
            rows_failed += 1
            # Try to extract context for better error message
            try:
                row_chap_id = str(row.get("chap_id", "unknown")) if pd.notna(row.get("chap_id")) else "unknown"
                row_section_num = row.get("section_num", "unknown")
                row_parent_id = row.get("parent_section_id", "unknown")
                row_level = row.get("level", "unknown")
            except:
                row_chap_id = "unknown"
                row_section_num = "unknown"
                row_parent_id = "unknown"
                row_level = "unknown"
            
            error_msg = (
                f"Error processing row {index}: {type(e).__name__}: {str(e)}\n"
                f"  Context: section_num={row_section_num}, level={row_level}, parent_section_id={row_parent_id}, chap_id={row_chap_id}\n"
                f"  This row will be skipped."
            )
            raise ValueError(error_msg) from e

        
    output_df = pd.DataFrame(processed_rows)
    
    # Processing summary
    total_output_rows = len(processed_rows)
    parent_rows = sum(1 for r in processed_rows if not r['is_subsection'])
    subsection_rows = total_output_rows - parent_rows
    print(f"Processing Summary:")
    print(f"  Rows processed: {rows_processed}/{len(df)}")
    print(f"  Parent sections (level=0): {parent_rows}")
    print(f"  Sub-sections (level>0): {subsection_rows}")
    print(f"  Total output rows: {total_output_rows}")
    
    if output_file:
        output_df.to_csv(output_file, index=False)
        print(f"Output saved to {output_file}")
    
    return output_df