# com.microsoft::Inverse v1

The provider accepts exactly one present FLOAT or DOUBLE tensor X and one output
of the same type and shape. Rank must be known and at least two; the last two
dimensions must be square. Batch dimensions and matrix order may be symbolic and
change between calls. There are no attributes. Unknown rank, non-square matrices,
extra/missing slots and incompatible output geometry/types are declined.

## Execution and errors

Nonempty matrices use `mlx_linalg_inv` on the **actual run thread's default MLX
CPU stream**, including FLOAT inputs. This is provider-owned MLX/LAPACK execution,
not GPU acceleration or ORT CPU fallback. The context and its other operations
keep their existing stream policy; an F32 control-flow body is not moved wholesale
to CPU. Recursive eager boundaries prevent capture of Inverse or its parents.
The existing F64 plan policy and DOUBLE-body payload rejection remain unchanged.
Live host access completes input producers before inversion. Arrays are retained
in the context arena, and only a completed result is bound for surrounding consumers.
The existing default CPU wrapper is obtained inside the handler on the actual
run-owner thread, never cached from an EP-construction thread. Core MLX defaults
are thread-local per device and reused across sessions on that thread. The local
C handle lives through evaluation/draining and is then freed. This avoids a new
native queue/worker per plan: MLX retains explicit new streams and their workers
for process lifetime, so a dedicated stream per new session would accumulate
resources even after its C handle was freed. Native worker growth from thread
churn is not solved here.

This is **not exclusive queue ownership**. The existing run gate serializes a
plan and enforces its owner thread; unrelated asynchronous external MLX work on
that same thread may share its CPU error queue and is an unsupported/unverified
integration boundary. No preexisting stream error is silently cleared or retried.
Cross-thread behavior requires the bounded hosted regression described below,
not an all-concurrency guarantee.
MLX's existing cross-stream array dependencies handle subsequent GPU consumption;
no default stream is mutated and no captured placeholder is evaluated.

The handler validates live shape, checked storage and input finiteness on every
call, including intermediates and initializers. **NaN/Inf input is an explicit
runtime policy limit**, not a schema attribute or an asserted CUDA restriction.
It returns an EP error rather than defining a nonfinite matrix inverse.
Singular factorization, inversion and evaluation errors are reported without
retrying on another provider, returning a pseudoinverse, or replacing the result.
Ill-conditioned invertible inputs are not rejected by a guessed condition-number
threshold; their accuracy remains limited by the input dtype and LAPACK.
Overflow in an otherwise finite inversion is not silently clamped or widened.

The native result is evaluated and the selected CPU stream is synchronized before
binding it. Synchronization also runs after primitive construction/evaluation
failure, preserving the first nonzero status. This catches queued errors even
when an already-signaled array event bypasses its error check. Source tracing
supports this error path; actual singular-error/recovery acceptance requires the
hosted fresh-subprocess fixture to exit successfully after an exact EP error and
a valid subsequent call on the same session.

Empty batches and 0x0 matrices preserve their exact empty output shape without
calling the native kernel, which otherwise divides by N*N. This includes shapes
such as [0,50000,50000]: an unused N*N must not cause rejection. Every extent must
fit int32; nonempty element counts must fit int32 and byte counts must fit the
existing checked-layout bounds. The nonempty bound also protects the kernel's
signed-int N*N arithmetic. These checks are not a guarantee that an allocation
fits available RAM. No new solver, tiling, dependency or CPU-policy framework is
introduced.

Zero-trip Scan already translates a dummy body solely to derive scan-output
shape templates. An explicit saved/restored context scope marks only that path.
Inside it, Inverse returns a same-shape typed zero **template without reading
values or invoking inversion**. This is not a numerical inverse, an actual body
iteration, a replacement for any real input, or a singular-input fallback.
The template scope propagates through nested bodies and restores its previous
value on success and error; other body errors remain errors. Initial Scan states
are unchanged and scan outputs retain their zero-length sequence dimension.

## Authority and coverage limits

The retained authorized ORT v1 schema specifies homogeneous FLOAT16/FLOAT/DOUBLE
batched square inversion. **FLOAT16 remains an OPEN feasible promotion-path
candidate**, not an impossible operator form. This handler initially supports
FLOAT/DOUBLE because the pinned MLX primitive accepts those exact types.

Implementation evidence is pinned MLX
[`36969c09`](https://github.com/ml-explore/mlx/tree/36969c094354086e15d2a327a9c50feb20b9cf11):
`mlx/linalg.cpp:14-20,34-40,314-339` (CPU, dtype, shape and primitive),
`mlx/backend/cpu/inverse.cpp:11-68,106-160` (typed getrf/getri and kernel errors),
`mlx/backend/cpu/encoder.h:43-57` and `mlx/scheduler.cpp:101-145`
(task exceptions and event propagation), `mlx/error.h:37-42` (clear-and-throw),
`mlx/event.h:35-38`, and `mlx/backend/metal/event.cpp:45-48` (event error checks).
Pinned mlx-c
[`57731b00`](https://github.com/ml-explore/mlx-c/tree/57731b00e8bed710e0750e7356da9f18f8eb3e20):
`mlx/c/linalg.h:65`, `linalg.cpp:137-148`, `array.cpp:348-355` and
`stream.cpp:23-33,105-112,135-142` expose explicit stream creation and report
C++ exceptions. Core `mlx/stream.cpp:18-26,41-50` defines thread-local defaults
and their reuse; `71-82` retains explicit new stream indices.
`mlx/scheduler.cpp:50,149-172` creates workers lazily and retains the scheduler;
mlx-c `mlx/c/private/stream.h:12-22,43-54` documents owning handles, not native
worker reclamation.
These are retained source artifacts, not installed-symbol or binary-identity
proof, and do not establish ORT CUDA execution or bitwise equivalence.

Focused fixtures use independent NumPy inverses and two-sided residuals, manual
inverse goldens, pivoting, exact binary ill-conditioned examples and a genuine
DOUBLE exponent discriminator. They cover live repeated values, batches,
symbolic/empty/singleton geometry, initializer/intermediate inputs, strict
negative forms, explicit nonfinite errors/recovery, and isolated singular
errors/recovery. Nested runtime fixtures cover If depths 1/2, constant-trip Loop
and live sequence-length Scan (including zero, nested If, original-state
preservation and subsequent recovery); recursive descriptor fixtures cover their
fences. Template scope restoration has a focused Rust fixture. A fresh subprocess
also exercises simultaneous successful
and singular plans, requiring every numerical/error/recovery assertion and
exit0; this is bounded concurrency coverage, not proof of all possible schedules.
A separate fresh child repeatedly creates, runs and drops sessions on one
thread, then stays alive while macOS `leaks` scans its PID. The test requires an
explicit zero-leak report and a successful child exit. Keeping the final Python
process alive lets the scanner run after framework Python has finished launching.
A paired native fixture checks that the scanner distinguishes a freed allocation
from a lost 1 MiB allocation. The scan checks unreachable heap allocations.
Provider profiles are required to contain actual nonempty all-MLX kernel events;
they do not identify a physical CPU stream. Separate trace assertions cover eager
Inverse/If execution and compiled neighbor cache reuse, not CPU/GPU hardware
measurement. Performance and large-allocation feasibility remain unmeasured.
