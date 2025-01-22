"""Combine curated data from different datasets."""

import json

from datasets import concatenate_datasets, load_dataset
from util.prompt import SKY_T1_FIXED


def map_numina_conversations(x):
    """Map the Numina dataset to the required format."""
    user_message = f"Return your final response within \\boxed{{}}. {x['problem']}"
    assistant_message = (
        f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>"
    )
    return {
        "system": SKY_T1_FIXED,
        "conversations": [
            {"from": "user", "value": user_message},
            {"from": "assistant", "value": assistant_message},
        ],
    }


numina_rejection_correct = (
    load_dataset("bespokelabs/sky-t1-numina-rejection-sampled", trust_remote_code=True)["train"].filter(lambda x: x["correct"]).take(10_500)
)
numina_conversations = numina_rejection_correct.map(map_numina_conversations, remove_columns=numina_rejection_correct.column_names)


def map_apps_conversations(x):
    """Map the APPS dataset to the required format."""
    test_case = json.loads(x["input_output"])
    starter_code = x["starter_code"]
    prompt = x["question"]

    user_message = ""
    data = test_case
    if not data.get("fn_name"):
        user_message += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # "\nUse Standard Input format"#\n" #noqa
    else:
        user_message += "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # "\nUse Call-Based format"#\n" #noqa
    data = prompt
    user_message += data
    if starter_code is not None:
        data = starter_code
        data = "\n" + data
        user_message += data
    else:
        pass
    assistant_message = (
        f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>"
    )

    return {
        "system": SKY_T1_FIXED,
        "conversations": [
            {"from": "user", "value": user_message},
            {"from": "assistant", "value": assistant_message},
        ],
    }


apps_conversations_correct = load_dataset("bespokelabs/sky-t1-apps-rejection-sampled", trust_remote_code=True)["train"].filter(lambda x: x["correctness"])
apps_conversations = apps_conversations_correct.map(map_apps_conversations, remove_columns=apps_conversations_correct.column_names)


def map_taco_conversations(x):
    """Map the TACO dataset to the required format."""
    test_case = json.loads(x["input_output_x"])
    starter_code = x["starter_code"]
    prompt = x["question"]

    user_message = ""
    data = test_case
    if not data.get("fn_name"):
        user_message += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # "\nUse Standard Input format"#\n" #noqa
    else:
        user_message += "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # "\nUse Call-Based format"#\n" #noqa
    data = prompt
    user_message += data
    if starter_code is not None:
        data = starter_code
        data = "\n" + data
        user_message += data
    else:
        pass
    assistant_message = (
        f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>"
    )

    return {
        "system": SKY_T1_FIXED,
        "conversations": [
            {"from": "user", "value": user_message},
            {"from": "assistant", "value": assistant_message},
        ],
    }


taco_train_data_correct = load_dataset("bespokelabs/sky-t1-taco-train-rejection-sampled-shreyas", trust_remote_code=True)["train"].filter(
    lambda x: x["correctness"]
)
taco_train_conversations = taco_train_data_correct.map(map_taco_conversations, remove_columns=taco_train_data_correct.column_names)
taco_train_conversations.push_to_hub(
    "bespokelabs/sky-t1-taco-train-rejection-sampled-shreyas-conversations",
    private=True,
)

taco_test_data_correct = load_dataset("bespokelabs/sky-t1-taco-test-rejection-sampled-shreyas", trust_remote_code=True)["train"].filter(
    lambda x: x["correctness"]
)
taco_test_conversations = taco_test_data_correct.map(map_taco_conversations, remove_columns=taco_test_data_correct.column_names)
taco_test_conversations.push_to_hub("bespokelabs/sky-t1-taco-test-rejection-sampled-shreyas-conversations", private=True)


def map_still2(x):
    """Map the still2 dataset to the required format."""
    return {
        "system": SKY_T1_FIXED,
        "conversations": [
            {"from": "user", "value": x["question"]},
            {"from": "assistant", "value": x["combined_text"]},
        ],
    }


still2 = load_dataset("RUC-AIBOX/long_form_thought_data_5k", trust_remote_code=True)["train"]
still2_columns = still2.column_names
still2 = still2.filter(lambda x: x["domain"] in ["puzzle", "physics", "biology", "chemistry"]).map(map_still2, remove_columns=still2_columns)


def validate_row(row, idx):
    """Validates a single row of the dataset.

    Args:
        row: Dataset row to validate
        idx: Index of the row for error reporting

    Returns:
        list: List of error messages if any were found, empty list if valid
    """
    errors = []

    # Check system field exists
    if "system" not in row:
        errors.append(f"Row {idx}: Missing 'system' field")
        return errors

    # Check conversations field exists and has correct length
    if "conversations" not in row:
        errors.append(f"Row {idx}: Missing 'conversations' field")
        return errors

    convs = row["conversations"]
    if len(convs) != 2:
        errors.append(f"Row {idx}: Expected 2 conversations, found {len(convs)}")
        return errors

    # Validate first message (user)
    if convs[0].get("from") != "user":
        errors.append(f"Row {idx}: First message must be from 'user', found '{convs[0].get('from')}'")
    if not convs[0].get("value"):
        errors.append(f"Row {idx}: User message value is empty")

    # Validate second message (assistant)
    if convs[1].get("from") != "assistant":
        errors.append(f"Row {idx}: Second message must be from 'assistant', found '{convs[1].get('from')}'")

    assistant_msg = convs[1].get("value", "")
    if not assistant_msg:
        errors.append(f"Row {idx}: Assistant message value is empty")
        return errors

    # Check for required tags in assistant message
    required_tags = [
        "<|begin_of_thought|>",
        "<|end_of_thought|>",
        "<|begin_of_solution|>",
        "<|end_of_solution|>",
    ]
    for tag in required_tags:
        if tag not in assistant_msg:
            errors.append(f"Row {idx}: Missing required tag '{tag}' in assistant message")

    return errors


def validate_final_dataset(dataset):
    """Validates that each row in the dataset meets the required format using map.

    Args:
        dataset: HuggingFace dataset to validate

    Returns:
        bool: True if dataset is valid, False otherwise
        list: List of error messages if any were found
    """
    validation_results = dataset.map(
        lambda x, idx: {"errors": validate_row(x, idx)},
        with_indices=True,
        remove_columns=dataset.column_names,
    )

    all_errors = []
    for result in validation_results:
        all_errors.extend(result["errors"])

    is_valid = len(all_errors) == 0
    return is_valid, all_errors


# Add validation before pushing to hub
final_dataset = concatenate_datasets(
    [
        numina_conversations,
        apps_conversations,
        still2,
        taco_train_conversations,
        taco_test_conversations,
    ]
)

# Validate the dataset
is_valid, validation_errors = validate_final_dataset(final_dataset)

if not is_valid:
    print("Dataset validation failed!")
    print("\n".join(validation_errors))
    raise ValueError("Dataset validation failed")

final_dataset.push_to_hub("bespokelabs/stratos-r1", private=True)
