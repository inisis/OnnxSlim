---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: inisis

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the bug:

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
run the script and paste the output.
```
python -c "import importlib; pkgs=['onnx','onnxruntime','onnxslim']; \
[print(f'{p}: {importlib.import_module(p).__version__}') if importlib.util.find_spec(p) else print(f'{p}: missing') for p in pkgs]"
```

**Additional context**
Add any other context about the problem here.
