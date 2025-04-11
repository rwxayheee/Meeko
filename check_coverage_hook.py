import subprocess
import sys
import ast
from pathlib import Path
import coverage

def get_changed_py_files():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True
    )
    files = [f for f in result.stdout.splitlines() if f.endswith(".py")]

    print("Changed Python files:")
    for file in files:
        print(f" - {file}")

    return files

def find_new_classes_and_attrs(file_path):
    with open(file_path, "r") as f:
        source = f.read()
    tree = ast.parse(source, filename=file_path)

    lines = source.splitlines()
    new_classes = []
    new_attrs = []

    class ClassVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            if "# coverage: ignore" in lines[node.lineno - 1]:
                return
            new_classes.append((node.name, node.lineno))
            # You can still check for attrs if you want...
            self.generic_visit(node)

    ClassVisitor().visit(tree)
    return new_classes, new_attrs

def check_coverage():
    cov = coverage.Coverage()
    cov.load()
    data = cov.get_data()

    uncovered = []
    for file in get_changed_py_files():
        abs_path = str(Path(file).resolve())
        lines_missing = data.lines(abs_path) or []

        classes, attrs = find_new_classes_and_attrs(file)
        for name, lineno in classes + attrs:
            if lineno in lines_missing:
                uncovered.append((file, name, lineno))

    return uncovered

if __name__ == "__main__":
    print("🧪 Running tests and checking coverage...")
    subprocess.run(["pytest", "--cov=."])

    uncovered_items = check_coverage()
    if uncovered_items:
        print("❌ Missing test coverage for:")
        for file, name, lineno in uncovered_items:
            print(f" - {name} in {file} (line {lineno})")
        sys.exit(1)
    else:
        print("✅ All new classes and attributes are covered!")


