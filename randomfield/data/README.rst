Data directory
==============

This directory contains data files included with the affiliated package source
code distribution. Note that this is intended only for relatively small files
- large files should be externally hosted and downloaded as needed.

The file ``default_power.dat`` contains a default power spectrum tabulated as
two columns, k in h/Mpc and P(k) in (Mpc/h)**3.  Use
:func:`randomfield.power.load_default_power` to read this file.
