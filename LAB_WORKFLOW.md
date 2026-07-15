# AI Lab Workflow Guide

For future labs, follow this exact workflow:

1. **Directory Setup**: Create a new top-level folder matching the lab number (e.g., `8/`).
2. **Copy Files**: Copy the corresponding lab materials from `guide/AI_Practical_ACEM/New_Syllabus/` into the new folder.
3. **Code Fixes**: Review the code for errors (like Windows-style backslashes in paths) and fix them so they run seamlessly on Linux.
4. **Update README**: Update `README.md` to point to the new folder. Include a new sub-section featuring a runnable PyTorch code snippet with simulated console output.
5. **Inline Report**: DO NOT create a standalone report file. Instead, append a Markdown cell at the very end of the lab's `.ipynb` notebook containing the Lab Report. The report must include the following headers: 
   - 1. Objective
   - 2. Methodology
   - 3. Results
   - 4. Discussion
   - 5. Conclusion
