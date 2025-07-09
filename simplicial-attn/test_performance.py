#!/usr/bin/env python3
"""
Performance tests and benchmarks for 2-simplicial attention.

Benchmarks cover:
- Triton vs PyTorch reference performance
- Throughput measurements (FLOPS, tokens/sec)
- Memory usage analysis
- Scaling characteristics with sequence length, batch size, etc.
- Performance across different hardware configurations
"""

import torch
import time
import math
import gc

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .triton_impl import TwoSimplicialAttention
from .reference import two_simplicial_attention_reference


def scaled_dot_product_attention_reference(q, k, v, scale=None, causal=False):
    """
    Reference implementation of traditional scaled dot product attention.
    
    Args:
        q, k, v: [batch, seq_len, num_heads, head_dim]
        scale: Attention scale factor
        causal: Whether to apply causal masking
    
    Returns:
        output: [batch, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Compute attention scores: Q @ K^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, seq_len, num_heads, seq_len]
    
    # Apply causal mask if requested
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask[None, :, None, :], float('-inf'))
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply to values: Attention @ V
    output = torch.matmul(attn_weights, v)  # [batch, seq_len, num_heads, head_dim]
    
    return output


def scaled_dot_product_attention_optimized(q, k, v, scale=None, causal=False):
    """
    Optimized implementation using PyTorch's built-in scaled_dot_product_attention.
    
    Args:
        q, k, v: [batch, seq_len, num_heads, head_dim]
        scale: Attention scale factor
        causal: Whether to apply causal masking
    
    Returns:
        output: [batch, seq_len, num_heads, head_dim]
    """
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        # Use PyTorch's optimized implementation if available (PyTorch 2.0+)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=scale
        )
    else:
        # Fallback to reference implementation
        return scaled_dot_product_attention_reference(q, k, v, scale, causal)


class TraditionalAttention(torch.nn.Module):
    """PyTorch module wrapper for traditional scaled dot product attention."""
    
    def __init__(self, head_dim, scale=None, causal=False, use_optimized=True):
        super().__init__()
        self.head_dim = head_dim
        self.scale = scale if scale is not None else (1.0 / math.sqrt(head_dim))
        self.causal = causal
        self.use_optimized = use_optimized
    
    def forward(self, q, k, v):
        """
        Forward pass using traditional attention.
        
        Args:
            q, k, v: [batch, seq_len, num_heads, head_dim]
        
        Returns:
            output: [batch, seq_len, num_heads, head_dim]
        """
        if self.use_optimized:
            return scaled_dot_product_attention_optimized(q, k, v, self.scale, self.causal)
        else:
            return scaled_dot_product_attention_reference(q, k, v, self.scale, self.causal)


def get_gpu_memory_usage() -> Tuple[float, float]:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0.0, 0.0


def get_gpu_info() -> Dict[str, str]:
    """Get GPU device information."""
    if not torch.cuda.is_available():
        return {"device": "CPU only"}
    
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
    
    return {
        "device": f"GPU {device}",
        "name": gpu_name,
        "memory_gb": f"{gpu_memory:.1f}",
        "compute_capability": f"{torch.cuda.get_device_properties(device).major}.{torch.cuda.get_device_properties(device).minor}"
    }


@dataclass 
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    forward_time_ms: float
    backward_time_ms: Optional[float]
    total_time_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    throughput_tokens_per_sec: float
    throughput_gflops: Optional[float]
    config: Dict


class PerformanceBenchmark:
    """Performance benchmarking suite for 2-simplicial attention."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
    def warmup_gpu(self, num_iterations: int = 10):
        """Warm up GPU for stable measurements."""
        if self.device.type == 'cpu':
            return
            
        print("🔥 Warming up GPU...")
        warmup_tensor = torch.randn(1000, 1000, device=self.device)
        for _ in range(num_iterations):
            torch.matmul(warmup_tensor, warmup_tensor)
        torch.cuda.synchronize()
        
    def estimate_flops_simplicial(self, batch_size: int, seq_len: int, num_heads: int, head_dim: int, 
                                  w1: int, w2: int) -> float:
        """
        Estimate FLOPs for 2-simplicial attention.
        
        This is approximate since the actual computation depends on window overlap patterns.
        """
        # Forward pass FLOPs (rough estimate)
        # For each query position, we compute attention over w1 × w2 key-value pairs
        attention_pairs_per_query = min(w1, seq_len) * min(w2, seq_len)
        
        # Score computation: q @ (k1 * k2) for each pair
        score_flops = batch_size * seq_len * num_heads * attention_pairs_per_query * head_dim
        
        # Softmax: exp + normalization
        softmax_flops = batch_size * seq_len * num_heads * attention_pairs_per_query * 2
        
        # Value computation: weighted sum of (v1 * v2)
        value_flops = batch_size * seq_len * num_heads * attention_pairs_per_query * head_dim * 2
        
        total_flops = score_flops + softmax_flops + value_flops
        
        # Backward pass is roughly 2-3x forward pass
        return total_flops * 3.0
    
    def estimate_flops_traditional(self, batch_size: int, seq_len: int, num_heads: int, head_dim: int) -> float:
        """
        Estimate FLOPs for traditional scaled dot product attention.
        
        Standard attention: O(n^2 * d) complexity
        """
        # Forward pass FLOPs
        # Q @ K^T: [batch, seq, heads, dim] @ [batch, seq, heads, dim]^T 
        qk_flops = batch_size * num_heads * seq_len * seq_len * head_dim
        
        # Softmax: exp + normalization over seq_len dimension
        softmax_flops = batch_size * num_heads * seq_len * seq_len * 2
        
        # Attention @ V: [batch, seq, heads, seq] @ [batch, seq, heads, dim]
        av_flops = batch_size * num_heads * seq_len * seq_len * head_dim
        
        total_flops = qk_flops + softmax_flops + av_flops
        
        # Backward pass is roughly 2-3x forward pass
        return total_flops * 3.0
    
    def benchmark_implementation(self, impl_name: str, attention_func, 
                                batch_size: int, seq_len: int, num_heads: int, head_dim: int,
                                w1: int = None, w2: int = None, dtype: torch.dtype = torch.bfloat16, 
                                num_warmup: int = 10, num_iterations: int = 100,
                                test_backward: bool = True, attention_type: str = "simplicial") -> BenchmarkResult:
        """Benchmark a specific attention implementation."""
        
        # Create input tensors based on attention type
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       device=self.device, dtype=dtype, requires_grad=test_backward)
        
        if attention_type == "simplicial":
            k1 = torch.randn_like(q, requires_grad=test_backward)
            k2 = torch.randn_like(q, requires_grad=test_backward)
            v1 = torch.randn_like(q, requires_grad=test_backward)
            v2 = torch.randn_like(q, requires_grad=test_backward)
            inputs = [q, k1, k2, v1, v2]
        else:  # traditional attention
            k = torch.randn_like(q, requires_grad=test_backward)
            v = torch.randn_like(q, requires_grad=test_backward)
            inputs = [q, k, v]
        
        # Warmup
        for _ in range(num_warmup):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            output = attention_func(*inputs)
            
            if test_backward:
                grad_output = torch.randn_like(output)
                output.backward(grad_output, retain_graph=True)
                # Clear gradients
                for inp in inputs:
                    if inp.grad is not None:
                        inp.grad.zero_()
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Clear memory
        del output
        if test_backward:
            del grad_output
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        gc.collect()
        
        # Measure memory before benchmark
        mem_before_alloc, mem_before_reserved = get_gpu_memory_usage()
        
        # Benchmark forward pass
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        forward_start = time.perf_counter()
        for _ in range(num_iterations):
            output = attention_func(*inputs)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        forward_end = time.perf_counter()
        
        forward_time_ms = (forward_end - forward_start) * 1000 / num_iterations
        
        # Benchmark backward pass if requested
        backward_time_ms = None
        if test_backward:
            grad_output = torch.randn_like(output)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            backward_start = time.perf_counter()
            for _ in range(num_iterations):
                # Clear gradients first
                for inp in inputs:
                    if inp.grad is not None:
                        inp.grad.zero_()
                
                # Recompute output and backward
                output_bwd = attention_func(*inputs)
                output_bwd.backward(grad_output, retain_graph=True)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
            backward_end = time.perf_counter()
            
            backward_time_ms = (backward_end - backward_start) * 1000 / num_iterations
        
        # Measure memory after benchmark
        mem_after_alloc, mem_after_reserved = get_gpu_memory_usage()
        memory_usage_alloc = mem_after_alloc - mem_before_alloc
        memory_usage_reserved = mem_after_reserved - mem_before_reserved
        
        # Calculate throughput
        total_tokens = batch_size * seq_len * num_heads
        total_time_ms = forward_time_ms + (backward_time_ms or 0)
        throughput_tokens_per_sec = (total_tokens * 1000) / total_time_ms if total_time_ms > 0 else 0
        
        # Calculate GFLOPS based on attention type
        if attention_type == "simplicial":
            estimated_flops = self.estimate_flops_simplicial(batch_size, seq_len, num_heads, head_dim, w1, w2)
        else:  # traditional attention
            estimated_flops = self.estimate_flops_traditional(batch_size, seq_len, num_heads, head_dim)
        
        throughput_gflops = (estimated_flops / 1e9) / (total_time_ms / 1000) if total_time_ms > 0 else None
        
        config = {
            "batch_size": batch_size,
            "seq_len": seq_len, 
            "num_heads": num_heads,
            "head_dim": head_dim,
            "w1": w1,
            "w2": w2,
            "dtype": str(dtype),
            "test_backward": test_backward,
            "attention_type": attention_type
        }
        
        return BenchmarkResult(
            name=impl_name,
            forward_time_ms=forward_time_ms,
            backward_time_ms=backward_time_ms,
            total_time_ms=total_time_ms,
            memory_allocated_mb=memory_usage_alloc,
            memory_reserved_mb=memory_usage_reserved,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            throughput_gflops=throughput_gflops,
            config=config
        )
    
    def compare_implementations(self, batch_size: int, seq_len: int, num_heads: int, head_dim: int,
                               w1: int, w2: int, dtype: torch.dtype = torch.bfloat16, 
                               include_traditional: bool = True) -> Dict[str, BenchmarkResult]:
        """Compare 2-simplicial vs Traditional attention implementations."""
        
        results = {}
        
        print(f"📊 Benchmarking: batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim}, w=({w1},{w2})")
        
        # Benchmark 2-Simplicial Triton implementation
        def triton_func(q, k1, k2, v1, v2):
            attn = TwoSimplicialAttention(head_dim, w1, w2)
            return attn(q, k1, k2, v1, v2)
        
        print("  🔥 Benchmarking 2-Simplicial Triton implementation...")
        results["simplicial_triton"] = self.benchmark_implementation(
            "2-Simplicial Triton", triton_func, batch_size, seq_len, num_heads, head_dim, 
            w1, w2, dtype, attention_type="simplicial"
        )
        
        # Benchmark Traditional Attention (optimized)
        if include_traditional:
            def traditional_optimized_func(q, k, v):
                attn = TraditionalAttention(head_dim, use_optimized=True)
                return attn(q, k, v)
            
            print("  ⚡ Benchmarking Traditional Attention (optimized)...")
            results["traditional_optimized"] = self.benchmark_implementation(
                "Traditional Optimized", traditional_optimized_func, batch_size, seq_len, num_heads, head_dim,
                dtype=dtype, attention_type="traditional"
            )
            
            # Benchmark Traditional Attention (reference) for smaller sequences
            if seq_len <= 128:  # Reference implementation can be slow
                def traditional_reference_func(q, k, v):
                    attn = TraditionalAttention(head_dim, use_optimized=False)
                    return attn(q, k, v)
                
                print("  📚 Benchmarking Traditional Attention (reference)...")
                results["traditional_reference"] = self.benchmark_implementation(
                    "Traditional Reference", traditional_reference_func, batch_size, seq_len, num_heads, head_dim,
                    dtype=dtype, attention_type="traditional"
                )
            else:
                print("  ⏭️  Skipping Traditional Reference (too slow for large sequences)")
        
        # Benchmark 2-Simplicial Reference implementation (only for smaller sizes)
        if seq_len <= 64:  # Reference is too slow for larger sequences
            def simplicial_reference_func(q, k1, k2, v1, v2):
                # Convert to float32 for reference (better precision)
                return two_simplicial_attention_reference(
                    q.float(), k1.float(), k2.float(), v1.float(), v2.float(), w1, w2
                ).to(dtype)
            
            print("  📚 Benchmarking 2-Simplicial Reference implementation...")
            results["simplicial_reference"] = self.benchmark_implementation(
                "2-Simplicial Reference", simplicial_reference_func, batch_size, seq_len, num_heads, head_dim, 
                w1, w2, dtype, test_backward=False, attention_type="simplicial"  # Reference doesn't support autograd
            )
        else:
            print("  ⏭️  Skipping 2-Simplicial Reference (too slow for large sequences)")
        
        return results
    
    def scaling_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark scaling characteristics across different dimensions."""
        
        scaling_results = {
            "sequence_length": [],
            "batch_size": [],
            "num_heads": [],
            "head_dim": [],
            "window_size": []
        }
        
        print("📈 Running scaling benchmarks...")
        
        # Sequence length scaling
        print("\n🔍 Sequence length scaling...")
        base_config = {"batch_size": 2, "num_heads": 4, "head_dim": 64, "w1": 32, "w2": 32}
        for seq_len in [64, 128, 256, 512, 1024, 2048]:
            try:
                results = self.compare_implementations(seq_len=seq_len, **base_config)
                if "simplicial_triton" in results:
                    scaling_results["sequence_length"].append(results["simplicial_triton"])
            except Exception as e:
                print(f"    ⚠️  Failed at seq_len={seq_len}: {e}")
                break
        
        # Batch size scaling
        print("\n🔍 Batch size scaling...")
        base_config = {"seq_len": 256, "num_heads": 4, "head_dim": 64, "w1": 32, "w2": 32}
        for batch_size in [1, 2, 4, 8, 16, 32]:
            try:
                results = self.compare_implementations(batch_size=batch_size, **base_config)
                if "simplicial_triton" in results:
                    scaling_results["batch_size"].append(results["simplicial_triton"])
            except Exception as e:
                print(f"    ⚠️  Failed at batch_size={batch_size}: {e}")
                break
        
        # Number of heads scaling
        print("\n🔍 Number of heads scaling...")
        base_config = {"batch_size": 2, "seq_len": 256, "head_dim": 64, "w1": 32, "w2": 32}
        for num_heads in [1, 2, 4, 8, 16, 32]:
            try:
                results = self.compare_implementations(num_heads=num_heads, **base_config)
                if "simplicial_triton" in results:
                    scaling_results["num_heads"].append(results["simplicial_triton"])
            except Exception as e:
                print(f"    ⚠️  Failed at num_heads={num_heads}: {e}")
                break
        
        # Head dimension scaling
        print("\n🔍 Head dimension scaling...")
        base_config = {"batch_size": 2, "seq_len": 256, "num_heads": 4, "w1": 32, "w2": 32}
        for head_dim in [32, 64, 128, 256, 512]:
            try:
                results = self.compare_implementations(head_dim=head_dim, **base_config)
                if "simplicial_triton" in results:
                    scaling_results["head_dim"].append(results["simplicial_triton"])
            except Exception as e:
                print(f"    ⚠️  Failed at head_dim={head_dim}: {e}")
                break
        
        # Window size scaling
        print("\n🔍 Window size scaling...")
        base_config = {"batch_size": 2, "seq_len": 512, "num_heads": 4, "head_dim": 64}
        for window_size in [8, 16, 32, 64, 128, 256]:
            try:
                results = self.compare_implementations(w1=window_size, w2=window_size, **base_config)
                if "simplicial_triton" in results:
                    scaling_results["window_size"].append(results["simplicial_triton"])
            except Exception as e:
                print(f"    ⚠️  Failed at window_size={window_size}: {e}")
                break
        
        return scaling_results
    
    def precision_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Benchmark different precision levels."""
        print("\n🎯 Precision benchmarks...")
        
        config = {"batch_size": 4, "seq_len": 256, "num_heads": 8, "head_dim": 64, "w1": 32, "w2": 32}
        results = {}
        
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            print(f"  📊 Testing {dtype}...")
            try:
                impl_results = self.compare_implementations(dtype=dtype, **config)
                if "simplicial_triton" in impl_results:
                    results[str(dtype)] = impl_results["simplicial_triton"]
            except Exception as e:
                print(f"    ⚠️  Failed for {dtype}: {e}")
        
        return results
    
    def memory_benchmark(self) -> Dict[str, Dict]:
        """Benchmark memory usage patterns."""
        print("\n💾 Memory usage benchmarks...")
        
        memory_results = {}
        base_config = {"num_heads": 4, "head_dim": 64, "w1": 32, "w2": 32, "dtype": torch.bfloat16}
        
        # Test different sizes
        test_configs = [
            {"batch_size": 1, "seq_len": 512},
            {"batch_size": 4, "seq_len": 512}, 
            {"batch_size": 1, "seq_len": 1024},
            {"batch_size": 2, "seq_len": 1024},
            {"batch_size": 1, "seq_len": 2048},
        ]
        
        for config in test_configs:
            config_key = f"b{config['batch_size']}_s{config['seq_len']}"
            print(f"  📊 Testing {config_key}...")
            
            try:
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                gc.collect()
                
                mem_before = get_gpu_memory_usage()
                results = self.compare_implementations(**config, **base_config)
                mem_after = get_gpu_memory_usage()
                
                memory_results[config_key] = {
                    "config": config,
                    "memory_before_mb": mem_before,
                    "memory_after_mb": mem_after,
                    "memory_delta_mb": (mem_after[0] - mem_before[0], mem_after[1] - mem_before[1]),
                    "benchmark_result": results.get("triton")
                }
                
            except Exception as e:
                print(f"    ⚠️  Failed for {config_key}: {e}")
                memory_results[config_key] = {"error": str(e)}
        
        return memory_results
    
    def traditional_vs_simplicial_benchmark(self) -> Dict[str, List[Dict]]:
        """Benchmark comparing traditional vs 2-simplicial attention across different sequence lengths."""
        print("\n🔥 Traditional vs 2-Simplicial Attention Comparison")
        
        comparison_results = []
        base_config = {"batch_size": 2, "num_heads": 4, "head_dim": 64, "w1": 32, "w2": 32, "dtype": torch.bfloat16}
        
        for seq_len in [64, 128, 256, 512, 1024, 2048]:
            print(f"\n📊 Sequence Length: {seq_len}")
            
            try:
                results = self.compare_implementations(seq_len=seq_len, **base_config, include_traditional=True)
                
                # Extract key metrics for comparison
                comparison_data = {
                    "seq_len": seq_len,
                    "results": {}
                }
                
                for impl_name, result in results.items():
                    comparison_data["results"][impl_name] = {
                        "forward_time_ms": result.forward_time_ms,
                        "backward_time_ms": result.backward_time_ms,
                        "total_time_ms": result.total_time_ms,
                        "throughput_gflops": result.throughput_gflops,
                        "memory_allocated_mb": result.memory_allocated_mb,
                        "throughput_tokens_per_sec": result.throughput_tokens_per_sec
                    }
                
                # Calculate speedup ratios
                if "simplicial_triton" in results and "traditional_optimized" in results:
                    trad_time = results["traditional_optimized"].total_time_ms
                    simp_time = results["simplicial_triton"].total_time_ms
                    speedup = trad_time / simp_time if simp_time > 0 else 0
                    comparison_data["speedup_vs_traditional"] = speedup
                    
                    # Memory efficiency
                    trad_mem = results["traditional_optimized"].memory_allocated_mb
                    simp_mem = results["simplicial_triton"].memory_allocated_mb
                    memory_ratio = simp_mem / trad_mem if trad_mem > 0 else 0
                    comparison_data["memory_ratio_vs_traditional"] = memory_ratio
                    
                    print(f"   ⚡ 2-Simplicial vs Traditional: {speedup:.1f}x speedup, {memory_ratio:.1f}x memory")
                
                comparison_results.append(comparison_data)
                
            except Exception as e:
                print(f"    ⚠️  Failed at seq_len={seq_len}: {e}")
                break
        
        return {"comparison_by_seqlen": comparison_results}
    
    def generate_report(self, scaling_results: Dict, precision_results: Dict, memory_results: Dict, 
                       comparison_results: Dict = None):
        """Generate a comprehensive performance report."""
        
        print("\n" + "="*80)
        print("📋 PERFORMANCE REPORT")
        print("="*80)
        
        # System info
        gpu_info = get_gpu_info()
        print(f"\n🖥️  System Information:")
        for key, value in gpu_info.items():
            print(f"   {key}: {value}")
        
        # Precision comparison
        if precision_results:
            print(f"\n🎯 Precision Performance:")
            print(f"{'Precision':<12} {'Forward (ms)':<12} {'Backward (ms)':<12} {'Total (ms)':<12} {'GFLOPS':<10}")
            print("-" * 65)
            
            for precision, result in precision_results.items():
                forward_ms = f"{result.forward_time_ms:.2f}"
                backward_ms = f"{result.backward_time_ms:.2f}" if result.backward_time_ms else "N/A"
                total_ms = f"{result.total_time_ms:.2f}"
                gflops = f"{result.throughput_gflops:.1f}" if result.throughput_gflops else "N/A"
                
                print(f"{precision:<12} {forward_ms:<12} {backward_ms:<12} {total_ms:<12} {gflops:<10}")
        
        # Scaling characteristics
        for scale_type, results in scaling_results.items():
            if not results:
                continue
                
            print(f"\n📈 {scale_type.replace('_', ' ').title()} Scaling:")
            print(f"{'Parameter':<12} {'Forward (ms)':<12} {'Total (ms)':<12} {'Tokens/sec':<12} {'GFLOPS':<10}")
            print("-" * 65)
            
            for result in results:
                param_val = result.config[scale_type] if scale_type in result.config else "N/A"
                forward_ms = f"{result.forward_time_ms:.2f}"
                total_ms = f"{result.total_time_ms:.2f}"
                tokens_per_sec = f"{result.throughput_tokens_per_sec:.0f}"
                gflops = f"{result.throughput_gflops:.1f}" if result.throughput_gflops else "N/A"
                
                print(f"{param_val:<12} {forward_ms:<12} {total_ms:<12} {tokens_per_sec:<12} {gflops:<10}")
        
        # Memory usage
        if memory_results:
            print(f"\n💾 Memory Usage:")
            print(f"{'Config':<12} {'Allocated (MB)':<15} {'Reserved (MB)':<15} {'Performance':<15}")
            print("-" * 65)
            
            for config_name, data in memory_results.items():
                if "error" in data:
                    print(f"{config_name:<12} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15}")
                    continue
                    
                alloc_mb = f"{data['memory_delta_mb'][0]:.1f}"
                reserved_mb = f"{data['memory_delta_mb'][1]:.1f}"
                
                result = data.get('benchmark_result')
                if result:
                    perf = f"{result.total_time_ms:.1f}ms"
                else:
                    perf = "N/A"
                
                print(f"{config_name:<12} {alloc_mb:<15} {reserved_mb:<15} {perf:<15}")
        
        # Traditional vs Simplicial comparison
        if comparison_results and "comparison_by_seqlen" in comparison_results:
            print(f"\n🔥 Traditional vs 2-Simplicial Attention Comparison:")
            print(f"{'Seq Len':<10} {'2-Simp (ms)':<12} {'Trad (ms)':<12} {'Speedup':<10} {'2-Simp GFLOPS':<15} {'Trad GFLOPS':<15}")
            print("-" * 85)
            
            for data in comparison_results["comparison_by_seqlen"]:
                seq_len = data["seq_len"]
                results = data["results"]
                
                simp_time = results.get("simplicial_triton", {}).get("total_time_ms", 0)
                trad_time = results.get("traditional_optimized", {}).get("total_time_ms", 0)
                speedup = data.get("speedup_vs_traditional", 0)
                simp_gflops = results.get("simplicial_triton", {}).get("throughput_gflops", 0)
                trad_gflops = results.get("traditional_optimized", {}).get("throughput_gflops", 0)
                
                simp_time_str = f"{simp_time:.1f}" if simp_time else "N/A"
                trad_time_str = f"{trad_time:.1f}" if trad_time else "N/A"
                speedup_str = f"{speedup:.1f}x" if speedup else "N/A"
                simp_gflops_str = f"{simp_gflops:.1f}" if simp_gflops else "N/A"
                trad_gflops_str = f"{trad_gflops:.1f}" if trad_gflops else "N/A"
                
                print(f"{seq_len:<10} {simp_time_str:<12} {trad_time_str:<12} {speedup_str:<10} {simp_gflops_str:<15} {trad_gflops_str:<15}")
        
        print(f"\n{'='*80}")


def run_all_performance_tests():
    """Run all performance tests and generate report."""
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - performance tests require GPU")
        return
    
    print("🚀 Starting 2-Simplicial Attention Performance Benchmarks")
    print("="*70)
    
    benchmark = PerformanceBenchmark()
    benchmark.warmup_gpu()
    
    try:
        # Run different benchmark suites
        scaling_results = benchmark.scaling_benchmark()
        precision_results = benchmark.precision_benchmark()
        memory_results = benchmark.memory_benchmark()
        comparison_results = benchmark.traditional_vs_simplicial_benchmark()
        
        # Generate comprehensive report
        benchmark.generate_report(scaling_results, precision_results, memory_results, comparison_results)
        
        print("\n🎉 Performance benchmarks completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Performance benchmarks failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_performance_test():
    """Run a quick performance test for basic validation."""
    print("⚡ Quick Performance Test")
    print("-" * 30)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return False
    
    benchmark = PerformanceBenchmark()
    benchmark.warmup_gpu(num_iterations=3)
    
    # Test small configuration
    config = {"batch_size": 2, "seq_len": 128, "num_heads": 4, "head_dim": 64, "w1": 16, "w2": 16}
    
    try:
        results = benchmark.compare_implementations(**config, include_traditional=True)
        
        if "simplicial_triton" in results:
            simp_result = results["simplicial_triton"]
            print(f"✅ 2-Simplicial Triton Implementation:")
            print(f"   Forward: {simp_result.forward_time_ms:.2f} ms")
            print(f"   Backward: {simp_result.backward_time_ms:.2f} ms")
            print(f"   Total: {simp_result.total_time_ms:.2f} ms")
            print(f"   Throughput: {simp_result.throughput_tokens_per_sec:.0f} tokens/sec")
            if simp_result.throughput_gflops:
                print(f"   GFLOPS: {simp_result.throughput_gflops:.1f}")
        
        if "traditional_optimized" in results:
            trad_result = results["traditional_optimized"]
            print(f"⚡ Traditional Attention (Optimized):")
            print(f"   Forward: {trad_result.forward_time_ms:.2f} ms")
            print(f"   Backward: {trad_result.backward_time_ms:.2f} ms")
            print(f"   Total: {trad_result.total_time_ms:.2f} ms")
            print(f"   Throughput: {trad_result.throughput_tokens_per_sec:.0f} tokens/sec")
            if trad_result.throughput_gflops:
                print(f"   GFLOPS: {trad_result.throughput_gflops:.1f}")
            
            # Calculate speedup
            if "simplicial_triton" in results:
                speedup = trad_result.total_time_ms / simp_result.total_time_ms
                print(f"🚀 2-Simplicial vs Traditional: {speedup:.1f}x {'speedup' if speedup > 1 else 'slower'}")
        
        if "simplicial_reference" in results:
            ref_result = results["simplicial_reference"] 
            print(f"📚 2-Simplicial Reference Implementation:")
            print(f"   Forward: {ref_result.forward_time_ms:.2f} ms")
            if "simplicial_triton" in results:
                speedup = ref_result.forward_time_ms / simp_result.forward_time_ms
                print(f"   Triton Speedup: {speedup:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = quick_performance_test()
    else:
        success = run_all_performance_tests()
    
    exit(0 if success else 1) 