import re

def remove_footer(text: str) -> str:
    """
    Removes the footer from the given text.

    Args:
        text (str): The text containing the footer.

    Returns:
        str: The text with the footer removed.
    """
    footer_pattern = r"2025 Formula 1 Sporting Regulations\s+\d+/\d+\s+30 April 2025\s+©2025 Fédération Internationale de l’Automobile"
    cleaned_text = re.sub(footer_pattern, '', text)
    return cleaned_text