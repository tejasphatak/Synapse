// GELU activation function for GPT-2 FFN
// Implements: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// This is the "exact" GELU approximation used by GPT-2.

struct Params {
  total_elements: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608;  // sqrt(2/pi)
const COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.total_elements) {
    return;
  }

  let x = input[idx];
  let x3 = x * x * x;
  let inner = SQRT_2_OVER_PI * (x + COEFF * x3);

  // tanh approximation: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
  let e2x = exp(2.0 * inner);
  let tanh_val = (e2x - 1.0) / (e2x + 1.0);

  output[idx] = 0.5 * x * (1.0 + tanh_val);
}
