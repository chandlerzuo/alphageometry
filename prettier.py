import re
import random


# Replacement dictionary
replacement_dict = {
    '&': ['and', 'also', 'furthermore,', 'moreover,', 'in addition,'],
    '=': ['equals', 'is equal to', 'is equivalent to', 'is the same as', 'is identical to'],
    '⇒': ['implies', 'therefore', 'hence', '. This implies that', 'which means that'],
    '\\Delta': ['triangle', '△'],
    '∥': ['parallel to', 'is parallel to'],
    ':': [' isto ', ' over ', ' divided by ', ' to '],
    '∠': ['angle ', '∠'],
    '⟂': ['perpendicular to', 'is perpendicular to'],
}


def apply_replacements(input_statement, replacement_dict):
    # Function to replace a match with a random choice from the provided list of replacements
    def replace_with_random_choice(match):
        # The original text matched
        original_text = match.group(0)
        # Choose a random replacement for the matched text
        replacement = random.choice(replacement_dict[original_text])
        return replacement

    for find_text in replacement_dict.keys():
        # Escape special regex characters in find_text
        escaped_find_text = re.escape(find_text)
        # Compile a regex for the find_text
        regex = re.compile(escaped_find_text)
        # Replace occurrences of find_text with a randomly chosen replacement
        input_statement = regex.sub(replace_with_random_choice, input_statement)

    return input_statement


def translate_step(statement):
    # Helper function to replace & with a random conjunction
    # List of rules, each rule is a tuple (pattern, replacement)
    rules = [
        # Rule 0:        "XXX. <some statement 1> [XX] <some statement 2>" translate to
        # "From equation [XX], <some statement 1>, <some statement 2>"
        (r"^(\d{3})\. (.*?) \[(\d{2})\] (.*)", r"\1 - From equation \3, \2, \4"),

        # Rule 1:"<some statement 1> & <some statement 2> [XX] <some statement 3>" translate to
        #        "<some statement 1> & from [XX] <some statement 2> <some statement 3>"
        (r"(.*) & (.*) \[(\d{2})\] (.*)", r"\1 & from equation \3, \2 \4"),

        #  Rule 2: "<some statement 1> [XX]" translate to
        #          "<some statement 1>. Let's call this equation XX."
        (r"(.*) \[(\d{2})\]", r"\1. Let's call this equation \2."),
    ]

    # Process each rule
    for pattern, replacement in rules:
        # Check if the current pattern matches the statement
        while re.search(pattern, statement):
            # If it matches, apply the replacement
            statement = re.sub(pattern, replacement, statement)

    statement = apply_replacements(statement, replacement_dict)
    return statement


if __name__ == "__main__":
    # Proof steps from the problem
    proof_steps = [
        "001. D,A,F are collinear [00] & FA = FD [01] ⇒  F is midpoint of DA [06]",
        "002. D,E,G are collinear [02] & GD = GE [03] ⇒  G is midpoint of DE [07]",
        "003. F is midpoint of DA [06] & G is midpoint of DE [07] ⇒  FG ∥ AE [08]",
        "004. FG ∥ AE [08] & D,E,G are collinear [02] & D,A,F are collinear [00] ⇒  DG:GF = DE:EA [09]",
        "005. H,E,A are collinear [04] & HE = HA [05] ⇒  H is midpoint of AE [10]",
        "006. F is midpoint of DA [06] & H is midpoint of AE [10] ⇒  FH ∥ DE [11]",
        "007. FH ∥ DE [11] & H,E,A are collinear [04] & D,A,F are collinear [00] ⇒  DE:EA = HF:HA [12]",
        "008. DG:GF = DE:EA [09] & GD = GE [03] & DE:EA = HF:HA [12] ⇒  HF:HA = DG:GF",
        "001. DA = DB [00] & DI = DB [08] & DE = DB [01] ⇒  A,B,I,E are concyclic [10]",
        "002. A,B,I,E are concyclic [10] ⇒  ∠ABE = ∠AIE [11]",
        "003. FE = FB [02] & FH = FB [07] & FB = FC [03] & FG = FB [05] ⇒  B,G,H,E are concyclic [12]",
        "004. B,G,H,E are concyclic [12] ⇒  ∠BHG = ∠BEG [13]",
        "005. DA ⟂ HA [06] & AG ⟂ AD [04] & ∠ABE = ∠AIE [11] & G,I,E are collinear [09] & ∠BHG = ∠BEG [13] ⇒  ∠BHA = ∠BAI [14]",
        "006. DI = DB [08] & DA = DB [00] ⇒  D is the circumcenter of \Delta AIB [15]",
        "007. D is the circumcenter of \Delta AIB [15] & AG ⟂ AD [04] ⇒  ∠BAG = ∠BIA [16]",
        "008. DA ⟂ HA [06] & AG ⟂ AD [04] & ∠BAG = ∠BIA [16] ⇒  ∠BAH = ∠BIA [17]",
        "009. ∠BHA = ∠BAI [14] & ∠BAH = ∠BIA [17] (Similar Triangles)⇒  AB:IB = AH:IA",
    ]

    # Translate each proof step
    translated_steps = [translate_step(step) for step in proof_steps]

    # Output the translated proof
    for step in translated_steps:
        print(step)
