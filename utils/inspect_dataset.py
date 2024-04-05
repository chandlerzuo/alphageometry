import csv
import random


def load_data_from_csv(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='#')
        for row in reader:
            data.append(row)
    return data


def display_line(row):
    for key, value in row.items():
        if key != 'rnd_states':  # Exclude 'rnd_states' field
            print(f"{key}: {value}")
    print()


def main():
    filename = '../../datasets/nl_fl_dataset.csv'
    data = load_data_from_csv(filename)
    total_lines = len(data)
    current_line = random.randint(0, total_lines - 1)

    while True:
        print(f"Current line: {current_line + 1} / {total_lines}")
        display_line(data[current_line])

        user_input = input("Enter 'n' for next line, 'p' for previous line, or line number to display that line: ")
        if user_input == 'n':
            current_line = (current_line + 1) % total_lines
        elif user_input == 'p':
            current_line = (current_line - 1) % total_lines
        elif user_input.isdigit():
            line_number = int(user_input) - 1
            if 0 <= line_number < total_lines:
                current_line = line_number
            else:
                print("Invalid line number. Please enter a valid line number.")
        else:
            print("Invalid input. Please enter 'n' for next line, 'p' for previous line, or line number.")


if __name__ == "__main__":
    main()
