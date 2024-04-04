def transform_text(input_text):
    """
    Transforms the input text according to specified rules:
    - Replaces 'cong' with 'equal_seg'
    - Formats the output for point definitions and constraints
    """
    # Dictionary to hold equivalences between keywords
    equivalences = {
        "cong": "equal_seg",
        "perp": "on_tline",
        "para": "on_pline"
    }

    # Split the input text into lines
    lines = input_text.strip().split('\n')

    # Initialize an empty list to hold the output lines
    output_lines = []

    for line in lines:
        if line.endswith(": Points"):
            # Process point definitions
            points = line.replace(": Points", "").strip().split(' ')
            output_lines.extend([f"{point} = free {point};" for point in points])
        else:
            # Process constraints
            parts = line.split(' ')
            if parts[0] in equivalences:
                # Identify the replacement keyword
                keyword = equivalences[parts[0]]
                # Initialize the formatted line with the first argument and the keyword
                formatted_line = f"{parts[1]} = {keyword}"
                # Append each argument to the formatted line
                for part in parts[1:]:
                    if part.startswith('[') and part.endswith(']'):
                        break
                    formatted_line += f" {part}"
                # Finalize the formatted line
                formatted_line += ";"
                output_lines.append(formatted_line)

    # Join the output lines into a single text block
    return ' '.join(output_lines)


if __name__ == '__main__':
    input_text = \
"""
B C A D: Points
cong A B B C [00]
cong D A D B [01]
cong D B D C [02]
perp A B B C [00]
para A D B C [01]
para A B C D [02]
eqangle A C A B C A C B [00]
"""

    output_text = transform_text(input_text)
    print(output_text)
