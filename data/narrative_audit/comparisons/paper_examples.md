# Curated Side-by-Side Examples for SIGCSE Paper

Selection criteria: (1) full condition summary is clearly richer and more specific,
(2) full condition prediction was reasonably accurate, (3) the exercise is
understandable to a general reader, (4) the no_trace summary is notably more
generic. Example 4 is an intentional failure case for intellectual honesty.

Source data: v6.1 batch runs with GPT-5.4 (medium reasoning), 117 matched
transitions across three conditions (full trace, no trace, shuffled trace).

---

## Example 1: Letter-Category Counter (C/P Swap)

**Transition ID:** `594:5889:1649:9676:1`

### Exercise description

The student is building a program that reads a comma-separated string of letters
(e.g., `A,V,V,P,C,A,C,P,P,M`), counts occurrences of five categories (C, P, M,
V, A), finds the maximum count, and prints the result as a formatted vector. The
code already passes 1 of 3 tests. The two failing tests show that the counts for
C and P are swapped -- the vector's first two positions are reversed.

```python
vetor = [0,0,0,0,0]
zoio = input("string dos zoio:").split(',')
for i in range(len(zoio)):
    if zoio[i].upper() == "C":
        vetor[0] += 1       # <-- maps C to position 0
    elif zoio[i].upper() == "P":
        vetor[1] += 1       # <-- maps P to position 1
    # ... M -> 2, V -> 3, A -> 4
```

Failing test: input `A,V,V,P,C,A,C,P,P,M` expects `[3 2 1 2 2]` but gets
`[2 3 1 2 2]` -- the C count (3) and P count (2) are in the wrong positions.

### Full condition (with behavioral traces)

> **Student state summary:** Most evidence points to a small semantic swap fix.
> The current failures are exactly a C-versus-P reversal, and earlier in the same
> attempt the student corrected an A/V reversal by editing the compared letters
> rather than redesigning the program.

**Predicted events:**
1. `change` at line 5 -- change "C" to "P"
2. `change` at line 7 -- change "P" to "C"
3. `saida_testar` (run) -- run after C/P swap
4. `submit` -- submit if counts match

**Estimated probability:** 0.40

### No-trace condition (code + tests only)

> **Student state summary:** Likely infers the counting works and only the first
> two output positions are reversed.

**Predicted events:**
1. `idle_gap` -- pauses to compare expected and actual vectors
2. `change` at line 17 -- swap first two values in final print
3. `saida_testar` -- rerun tests after changing output order
4. `submit`

**Estimated probability:** 0.50

### What actually happened

| Metric | Full | No-trace |
|--------|------|----------|
| First event correct | Yes (change) | No (predicted idle_gap) |
| Episode motif correct | Yes (edit->run->submit) | No (predicted pause->edit->run->submit) |
| Event type edit similarity | 0.667 | 0.500 |
| Event type Jaccard overlap | 0.667 | 0.429 |

### Why this is compelling

The full condition uses the student's prior within-attempt behavior (correcting
an A/V reversal earlier) to infer they will swap the letter comparisons in the
if-conditions (lines 5 and 7), not rearrange the output format. The no-trace
condition, lacking this behavioral history, incorrectly predicts the student will
swap values at the *print* statement (line 17) -- a surface-level fix that would
not actually solve the bug. This demonstrates trace-grounded belief attribution:
the model reads behavioral patterns to infer *how* a student approaches fixes,
not just *what* is wrong.

---

## Example 2: Template Paste Detection (Wrong Category Labels)

**Transition ID:** `593:5886:6354:9710:0`

### Exercise description

The student must count occurrences of four categories in a comma-separated input
string and print a count vector. The expected categories are R, S, F, T (based
on the test inputs). However, the student's code checks for A, B, O, AB -- which
are blood type labels, clearly from a different exercise. All three tests produce
`[0 0 0 0]` because no input token matches any condition.

```python
cont = zeros(4, dtype=int)
vet = input().upper().split(",")
for i in vet:
    if i == "A":        # wrong label -- should be "R"
        cont[0] += 1
    elif i == "B":      # wrong label -- should be "S" or "F"
        cont[1] += 1
    elif i == "O":      # wrong label
        cont[2] += 1
    elif i == "AB":     # wrong label
        cont[3] += 1
print(cont)
```

### Full condition (with behavioral traces)

> **Student state summary:** A single full-code paste after a 56-second idle
> gap, no prior history, no local tests, and all-zero outputs suggest the student
> likely used a template from a different exercise and will next try the most
> obvious patch: change the four string comparisons to the exercise's actual
> categories, then submit again.

**Predicted events:**
1. `change` at line 4 -- replace first compared label
2. `change` at line 6 -- replace second compared label
3. `change` at line 8 -- replace third compared label
4. `change` at line 10 -- replace fourth compared label
5. `submit` -- resubmit after label edits

**Estimated probability:** 0.58

### No-trace condition (code + tests only)

> **Student state summary:** The student likely notices that none of the input
> tokens match the current if and elif cases, and makes a direct edit in the loop
> before testing again.

**Predicted events:**
1. `change` at lines 4-10 -- replace A B O AB checks with input symbols
2. `saida_testar` (run) -- run the program after updating comparisons

**Estimated probability:** 0.62

### What actually happened

| Metric | Full | No-trace |
|--------|------|----------|
| First event correct | Yes (change) | Yes (change) |
| Episode motif correct | Yes (edit->submit) | No (predicted edit->run) |
| Event type edit similarity | 0.417 | 0.083 |
| Event type Jaccard overlap | 0.417 | 0.077 |

### Why this is compelling

Both conditions identify the label mismatch, but the full condition produces a
qualitatively different cognitive attribution. It identifies the *behavioral
origin* of the bug -- a template paste from a different exercise -- by reading
the 56-second idle gap followed by a single full-code insertion with no local
testing. It also correctly predicts line-by-line edits (4 separate changes) and
direct submission without testing, while the no-trace condition collapses the
repair into one edit event and incorrectly predicts an intermediate test run.
The edit similarity (0.417 vs 0.083) reflects this granularity difference.

---

## Example 3: Array Transformation (Nonzero Decrement)

**Transition ID:** `591:5902:2844:9779:0`

### Exercise description

The student must transform a NumPy array: zeros become 9, nonzero values become
(value - 1). The code already handles zeros correctly (the `[0,0,0]` test
passes), but copies nonzero values unchanged instead of decrementing them.

```python
from numpy import *
v = array(eval(input()))
z = zeros(size(v), dtype="int")
for i in range(size(v)):
    if v[i] == 0:
        z[i] = 9           # correct
    else:
        z[i] = v[i]        # bug: should be v[i] - 1
print(z)
```

Test: input `[3,5,6,1,2,0]` expects `[2 4 5 0 1 9]` but gets `[3 5 6 1 2 9]`.

### Full condition (with behavioral traces)

> **Student state summary:** Student appears to have the loop, indexing, zero
> case, and output format working, but not the nonzero transformation. The failed
> tests differ only because nonzero values were copied unchanged.

**Predicted events:**
1. `change` at line 8 -- change copy to minus 1
2. `submit` -- resubmit after small fix

**Estimated probability:** 0.60

### No-trace condition (code + tests only)

> **Student state summary:** The student likely infers that nonzero elements
> should be decreased by one while keeping zeros mapped to nine.

**Predicted events:**
1. `idle_gap` -- pause to compare expected and actual outputs
2. `change` at line 8 -- change else branch to subtract one
3. `saida_testar` (run) -- run the program on sample inputs
4. `submit`

**Estimated probability:** 0.68

### What actually happened

| Metric | Full | No-trace |
|--------|------|----------|
| First event correct | Yes (change) | No (predicted idle_gap) |
| Episode motif correct | Yes (edit->submit) | No (predicted pause->edit->run->submit) |
| Event type edit similarity | 0.500 | 0.500 |
| Event type Jaccard overlap | 0.500 | 0.333 |

### Why this is compelling

Both conditions identify the correct bug (missing `- 1`) and predict the right
line. However, the full condition produces a tighter behavioral forecast: the
student will edit and submit directly (2 events), while the no-trace condition
predicts a more cautious sequence with a pause and intermediate test run (4
events). The actual student did exactly what the full condition predicted --
edited line 8 and submitted immediately. The full condition's summary also
demonstrates structural decomposition: it explicitly enumerates which parts of
the solution *work* (loop, indexing, zero case, output format) before isolating
the single failing component, showing the model's ability to reason about partial
correctness rather than treating the program as globally broken.

---

## Example 4 (Failure): Trace-Induced Overconfidence in Pause Prediction

**Transition ID:** `597:5849:1605:9794:0`

### Exercise description

The student is implementing a ring-value calculator using NumPy arrays. Each
ring value (1, 2, 3, 4) applies a different multiplier to a running product
starting at 200. The code is missing the handling for value 2 (should multiply
by 2), has a bug on the value-3 branch (`x[1]` instead of `x[i]`), and the
value-3 case does nothing (`p=p`).

```python
from numpy import *
x = array(eval(input("aneis: ")))
p = 200
i = 0
while i < size(x):
    if x[i] == 1:
        p = p * 4
    if x[1] == 3:      # bug: x[1] not x[i]; also p=p is a no-op
        p = p
    if x[i] == 4:
        p = p / 2
    i = i + 1
print(round(p, 2))
```

All 3 tests fail. Input `[3,2,2,2,4,4]` expects `400.0`, gets `50.0`.

### Full condition (with behavioral traces)

> **Student state summary:** Likely to focus on the specific branch causing every
> failed case with 3 and patch the two nearby lines rather than rewrite the whole
> program.

**Predicted events:**
1. `idle_gap` -- checks failed cases against 3 branch
2. `change` at line 7 -- replace `x[1]` with `x[i]`
3. `change` at line 8 -- change `p=p` to multiply by 8
4. `submit` -- rerun after local fix

**Estimated probability:** 0.62

### No-trace condition (code + tests only)

> **Student state summary:** The student likely compares expected outputs to
> current behavior and infers that value 2 needs its own multiplier rule.

**Predicted events:**
1. `change` at lines 7-8 -- add a case where value 2 doubles p
2. `submit` -- submit after adding the missing value 2 rule

**Estimated probability:** 0.62

### What actually happened

| Metric | Full | No-trace |
|--------|------|----------|
| First event correct | No (predicted idle_gap) | Yes (change) |
| Episode motif correct | No (predicted pause->edit->submit) | Yes (edit->submit) |
| Event type edit similarity | 0.429 | 0.286 |
| Event type Jaccard overlap | 0.375 | 0.286 |

Full condition composite accuracy: **0.201** vs no-trace: **0.643**

### Why this failure is informative

The full condition had access to the student's behavioral trace and used it to
construct a more detailed narrative -- focusing on the `x[1]` indexing bug in the
value-3 branch. But it over-interpreted the trace by predicting the student would
first *pause* to analyze the failures before editing, when the student actually
started editing immediately. The no-trace condition, with less information to
over-fit to, made the simpler (and correct) prediction that the student would
notice the missing value-2 case and jump straight to editing.

This illustrates a systematic pattern: across 117 transitions, the full condition
predicted `idle_gap` as the first event 19 times (16.2%) and was wrong in all 19
cases (100% false positive rate). In 16 of those 19 cases, the student actually
began editing immediately. By contrast, the no-trace condition predicted
`idle_gap` only 4 times (3.4%). Having richer behavioral context led the model to
over-attribute deliberative pausing -- projecting a more reflective process than
the student actually exhibited. This is a concrete example of trace-induced
overconfidence: more information does not always yield better predictions.
