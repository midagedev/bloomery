# q3k-gemv

Scaffolded cuda-oxide project.

## Setup

```bash
cargo oxide doctor
```

Fix anything doctor reports before building.

## Run

```bash
cargo oxide run
```

The template is a vector-add kernel. It uses `#[launch_contract]` and
`PreparedLaunch` so geometry is checked before launch. See the
cuda-oxide book getting-started chapter for the next steps.
