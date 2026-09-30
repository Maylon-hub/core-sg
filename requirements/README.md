# Qualified environment records

The `*-win-py311.txt` files record the Windows CPython 3.11 release qualification
environments, not universal platform locks. Build tools include the exact Cython,
NumPy, setuptools and wheel versions used. The runtime record covers the paired
MustaCHE environment. Install the generated CORE-SG artifact before MustaCHE;
do not use editable installs or PYTHONPATH. Preserve dependency downloads/hashes
for replay. Native compiler details remain part of the dated readiness report.
