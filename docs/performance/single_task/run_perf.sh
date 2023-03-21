#!/bin/bash
for i in {0..10..1}
do
echo "$i"

./.venv/bin/cssfinder --debug project ./5qubits_json/ run -t main_double; cat ./log/cssfinder | grep -Po "(?:Elapsed time: )([0-9]*\.?[0-9]*)" >> double_no_jit_uniform.txt

./.venv/bin/cssfinder --debug project ./5qubits_json/ run -t main_single; cat ./log/cssfinder | grep -Po "(?:Elapsed time: )([0-9]*\.?[0-9]*)" >> single_no_jit_uniform.txt

done
