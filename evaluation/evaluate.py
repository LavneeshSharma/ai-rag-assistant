from evaluation.test_cases import test_cases

from chains.rag_chain import create_rag_chain


def evaluate():

    total = len(test_cases)

    passed = 0

    for i, test in enumerate(test_cases):

        question = test["question"]

        expected_keywords = test["expected_keywords"]

        print(f"\n===== TEST {i+1} =====")
        print(f"Question: {question}")

        response = create_rag_chain(question)

        print("\nResponse:")
        print(response)

        success = True

        for keyword in expected_keywords:

            if keyword.lower() not in response.lower():
                success = False
                break

        if success:
            print("\nSTATUS: PASS")
            passed += 1
        else:
            print("\nSTATUS: FAIL")

    print("\n====================")
    print(f"FINAL SCORE: {passed}/{total}")


if __name__ == "__main__":
    evaluate()