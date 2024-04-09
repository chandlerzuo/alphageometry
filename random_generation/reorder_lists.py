def get_ordering_index(first_list, second_list):
    """
    Returns the indices to reorder the second list to match the order of the first list.
    """
    # Create a mapping from element to its index in the first list
    index_map = {element: index for index, element in enumerate(first_list)}

    # Create a list of tuples (index_in_first_list, index_in_second_list)
    ordering = sorted((index_map[element], index) for index, element in enumerate(second_list))

    # Extract and return just the indices from the second list in the correct order
    return [index for _, index in ordering]


if __name__ == "__main__":
    first_list = ['a', 'b', 'c', 'd']
    second_list = ['d', 'c', 'b', 'a']
    ordering_index = get_ordering_index(first_list, second_list)

    print("Ordering index:", ordering_index)
    print("Ordered second list:", [second_list[i] for i in ordering_index])
