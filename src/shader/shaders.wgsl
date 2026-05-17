@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn doubleMe(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    let array_length = arrayLength(&input);
    if(index >= array_length) {
        return;
    }

    output[index] = input[index] * 2.0;
}