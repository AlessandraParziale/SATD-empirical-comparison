import csv
import shutil
import os

CSV_PATH = "merged.csv"
BACKUP_PATH = "merged_backup.csv"
TMP_PATH = "merged_tmp.csv"

# Columns to remove (as requested)
COLUMNS_TO_REMOVE = [
    "FilePath", "ClassName", "MethodName", "Content", "CommentFor", "CommentsIn", "StartLine", "EndLine",
    "eachLabelCommentFor", "CommentForLabel", "eachLabelCommentsIn", "CommentsInLabel", "eachLabelCommentsAssociated",
    "PseudoLabelForCASFromMAT", "PseudoLabelForCASFromGGSATD", "PseudoLabelForCASFromXGBoost", "MethodSimplified",
    "class", "method", "constructor", "line", "cbo", "cboModified", "fanin", "fanout", "wmc", "rfc", "loc",
    "returnsQty", "variablesQty", "parametersQty", "methodsInvokedQty", "methodsInvokedLocalQty",
    "methodsInvokedIndirectLocalQty", "loopQty", "comparisonsQty", "tryCatchQty", "parenthesizedExpsQty",
    "stringLiteralsQty", "numbersQty", "assignmentsQty", "mathOperationsQty", "maxNestedBlocksQty",
    "anonymousClassesQty", "innerClassesQty", "lambdasQty", "uniqueWordsQty", "modifiers", "logStatementsQty",
    "hasJavaDoc"
]


def main():
    if not os.path.exists(CSV_PATH):
        print(f"{CSV_PATH} not found")
        return 1

    # Backup original
    shutil.copy2(CSV_PATH, BACKUP_PATH)

    with open(CSV_PATH, "r", newline="", encoding="utf-8-sig", errors="replace") as in_f:
        reader = csv.reader(in_f)
        try:
            header = next(reader)
        except StopIteration:
            print("Input CSV is empty")
            return 1

        # Determine indices to keep
        remove_set = set(COLUMNS_TO_REMOVE)
        keep_indices = [i for i, col in enumerate(header) if col not in remove_set]
        kept_header = [header[i] for i in keep_indices]

        # Write temporary file with filtered columns
        with open(TMP_PATH, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(kept_header)
            for row in reader:
                # pad or trim rows to header length if needed
                if not row:
                    continue
                # ensure row is at least as long as header
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                new_row = [row[i] for i in keep_indices]
                writer.writerow(new_row)

    # Replace original with filtered version
    shutil.move(TMP_PATH, CSV_PATH)
    print(f"Removed {len(remove_set & set(header))} columns. Backup saved to {BACKUP_PATH}.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
