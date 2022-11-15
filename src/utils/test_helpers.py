

def check_string_equality(string_one, string_two):
    assert string_one == string_two, \
        f"\n{string_one}\n" + \
        "="*100 + '\n' +\
        f"{string_two}"

def check_string_subset_of_another(string_one, string_two):
    assert string_one in string_two, \
        f"\n{string_one}\n" + \
        "="*100 + '\n' +\
        f"{string_two}"

def check_string_starts_with_another(string_one, string_two):
    assert string_one.startswith(string_two), \
        f"\n{string_one}\n" + \
        "="*100 + '\n' +\
        f"{string_two}"

def check_string_ends_with_another(string_one, string_two):
    assert string_one.endswith(string_two), \
        f"\n{string_one}\n" + \
        "="*100 + '\n' +\
        f"{string_two}"