@echo off

REM Generate the Sphinx documentation
sphinx-apidoc -o docs\source .\

REM Change directory to docs
cd docs

REM Build the HTML documentation
make html

REM Change back to the original directory
cd ..