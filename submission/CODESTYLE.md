# Code style: C++ / CUDA

These rules apply to all C++ (`.h`, `.cpp`) and CUDA (`.cuh`, `.cu`) code in this project, whether a
person or a coding agent (Claude Code, Codex, …) writes it. Agents: follow every rule below when you
create or edit code. If a rule can't be applied somewhere, say so in your report and explain why;
don't silently skip it.

## 1. What goes in headers

Headers (`.h`, `.cuh`) contain only:
- type definitions: `struct`, `class`, `enum`, `using`;
- function declarations;
- the definitions C++ requires in a header: `inline` and `__host__ __device__ inline` helpers used by
  more than one file, templates, and `constexpr` values.

Everything else (function bodies, kernels, globals) goes in a `.cpp` (host) or `.cu` (CUDA) file.

**CUDA exception: keep a kernel and its device code in one `.cu` file.** Every `__device__` function a
kernel calls must be defined in the same `.cu` file as that kernel. Don't split device code across
files: that needs separate device compilation (`-rdc`), which blocks inlining and changes the generated
GPU code. Other files call the kernel through a host launch function declared in the `.cuh` and defined
in the `.cu`.

## 2. Include guards, not `#pragma once`

```cpp
#ifndef PROJECT_RANGE_INDEX_H
#define PROJECT_RANGE_INDEX_H
// ...
#endif  // PROJECT_RANGE_INDEX_H
```

The guard name is a project prefix plus the file name in upper case, e.g. `U4M_RANGE_KERNEL_CUH`.

## 3. Brace initialisation for declarations

```cpp
int count{0};
double scale{1.5};
const uint32_t* base{v.encoded + offset};
```

- Braces catch narrowing conversions at compile time; build with `-Wall -Wextra -Wnarrowing`.
- **Container sizes keep parentheses.** `std::vector<int> v(n);` makes n elements, while
  `std::vector<int> v{n};` makes one element equal to n.
- Don't write `auto x{...}`. Use `auto x = f();` or spell out the type with braces.
- Later assignments are ordinary: `count = other;`.

## 4. One declaration per line

```cpp
// Good
int lo{0};
int hi{n};

// Not allowed
int lo{0}, hi{n};
int lo{0}; int hi{n};
```

This includes struct fields and arrays: one field or array per line.

```cpp
// Good
uint32_t lo[RBP_V3_UNROLL];
uint32_t hi[RBP_V3_UNROLL];
```

Exception: a short group of related values derived from built-ins, such as thread indices, may share
one `const` declaration with `=` initialisers, as long as the whole statement fits on one line:

```cpp
const uint32_t thread = threadIdx.x, threads = blockDim.x, lane = thread & 31, warp = thread >> 5, warps = threads >> 5;
```

## 5. No one-line bodies

Functions, loops, `if`/`else` and `switch` bodies always go on their own lines.

```cpp
// Good
void report(int value) {
    std::cout << value << '\n';
}

// Not allowed
void report(int value) { std::cout << value << '\n'; }
```

## 6. Braces on every body

Every loop and every `if` / `else` body has braces, even a single statement.

```cpp
// Good
while (lo < hi) {
    lo++;
}

// Not allowed
while (lo < hi) lo++;
```

## 7. Spacing

- After each `;` inside `for (...)`: `for (int i{0}; i < n; i++)`.
- Around binary and assignment operators: `a + b`, `x = y`, `lo < hi`.
- After commas: `f(a, b, c)`.
- Between a keyword and its parenthesis: `if (`, `for (`, `while (`, `switch (`.
- Pointers and references attach to the type: `const T* p`, `T& r`.

## 8. Formatter

Rules 5–7 are enforced by `clang-format` with this `.clang-format` (keep it at the code root). Lines are
at most 130 characters. A parameter list that doesn't fit on one line puts each parameter on its own
line, indented one level, with the closing parenthesis on its own line:

```cpp
void range_scores_launch(
    uint32_t ranges,
    uint32_t threads,
    // ...
    uint64_t* merge_keys
);
```

The configuration:

```yaml
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 130
AllowShortFunctionsOnASingleLine: None
AllowShortIfStatementsOnASingleLine: Never
AllowShortLoopsOnASingleLine: false
AllowShortBlocksOnASingleLine: Never
AllowShortLambdasOnASingleLine: None
AllowShortCaseLabelsOnASingleLine: false
SortIncludes: Never
InsertBraces: true
SpaceBeforeParens: ControlStatements
PointerAlignment: Left
AlignAfterOpenBracket: BlockIndent
BinPackParameters: OnePerLine
AllowAllParametersOfDeclarationOnNextLine: false
```

Run `clang-format -i` on every changed `.h/.cuh/.cpp/.cu` file before committing. Rules 1–4 are not
covered by the formatter and must be applied by hand.

## 9. Style changes must not change behaviour

- Make style-only changes in their own commit, separate from functional changes.
- For CUDA code, a style-only commit must leave the generated GPU code identical: compare the SASS
  before and after (`cuobjdump -sass`, built with the same toolkit and flags). If it differs, stop and
  report the difference instead of committing.
- Correctness checks (fixtures, exact comparison with the CPU reference) must still pass after any
  commit that touches code.

## Using this file with an agent

- **Claude Code:** add `@CODESTYLE.md` (with its path) to the project's `CLAUDE.md` or your own
  `CLAUDE.local.md`.
- **Codex:** reference it from `AGENTS.md`, e.g. "Follow the rules in CODESTYLE.md for all C++/CUDA
  code."
- In any prompt that edits code, it helps to add: "Follow CODESTYLE.md."
