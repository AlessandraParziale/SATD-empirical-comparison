import csv
import glob
import os
import sys

#!/usr/bin/env python3
"""
merge_dataset.py

Reads all CSV files in the current directory (except merged.csv) and merges them into merged.csv.
Assumes all CSVs share the same header.
"""

OUTPUT_NAME = "merged.csv"
# exact header as provided
EXPECTED_HEADER = "ID,FilePath,ClassName,MethodName,Content,CommentFor,CommentsIn,CommentsAssociated,StartLine,EndLine,eachLabelCommentFor,CommentForLabel,eachLabelCommentsIn,CommentsInLabel,eachLabelCommentsAssociated,CommentsAssociatedLabel,PseudoLabelForCASFromMAT,PseudoLabelForCASFromGGSATD,PseudoLabelForCASFromXGBoost,MethodSimplified,class,method,constructor,line,cbo,cboModified,fanin,fanout,wmc,rfc,loc,returnsQty,variablesQty,parametersQty,methodsInvokedQty,methodsInvokedLocalQty,methodsInvokedIndirectLocalQty,loopQty,comparisonsQty,tryCatchQty,parenthesizedExpsQty,stringLiteralsQty,numbersQty,assignmentsQty,mathOperationsQty,maxNestedBlocksQty,anonymousClassesQty,innerClassesQty,lambdasQty,uniqueWordsQty,modifiers,logStatementsQty,hasJavaDoc"
EXPECTED_HEADER_LIST = EXPECTED_HEADER.split(",")

def main():
    cwd = os.getcwd()
    csv_paths = sorted(p for p in glob.glob(os.path.join(cwd, "*.csv")) if os.path.basename(p) != OUTPUT_NAME)
    if not csv_paths:
        print("No CSV files found to merge.", file=sys.stderr)
        return 1

    with open(os.path.join(cwd, OUTPUT_NAME), "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(EXPECTED_HEADER_LIST)  # write header once

        for path in csv_paths:
            try:
                with open(path, "r", newline="", encoding="utf-8-sig", errors="replace") as in_f:
                    reader = csv.reader(in_f)
                    try:
                        header = next(reader)
                    except StopIteration:
                        # empty file, skip
                        continue
                    # If header doesn't match exactly, don't treat it as a data row
                    if header != EXPECTED_HEADER_LIST:
                        # If header looks like data (different), assume file still has header row and skip it
                        # (we won't validate further to avoid failing on minor differences)
                        pass
                    for row in reader:
                        if row:  # skip empty rows
                            writer.writerow(row)
            except Exception as e:
                print(f"Failed to process {path}: {e}", file=sys.stderr)

    print(f"Merged {len(csv_paths)} files into {OUTPUT_NAME}")
    return 0

if __name__ == "__main__":
    sys.exit(main())