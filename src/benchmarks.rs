use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

use crate::neural_net::EpochMetrics;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub epochs: Vec<usize>,
    pub losses: Vec<f32>,
    pub forward_times: Vec<Duration>,
    pub backward_times: Vec<Duration>,
    pub update_times: Vec<Duration>,
    pub memory_transfer_times: Vec<Duration>,
    pub host_computation_times: Vec<Duration>,
    pub total_times: Vec<Duration>,
    pub metadata: HashMap<String, String>,
}

impl BenchmarkResults {
    pub fn new() -> Self {
        Self {
            epochs: Vec::new(),
            losses: Vec::new(),
            forward_times: Vec::new(),
            backward_times: Vec::new(),
            update_times: Vec::new(),
            memory_transfer_times: Vec::new(),
            host_computation_times: Vec::new(),
            total_times: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn record_epoch(&mut self, epoch: usize, loss: f32, metrics: EpochMetrics) {
        self.epochs.push(epoch);
        self.losses.push(loss);
        self.forward_times.push(metrics.forward_time);
        self.backward_times.push(metrics.backward_time);
        self.update_times.push(metrics.update_time);
        self.memory_transfer_times.push(metrics.memory_transfers);
        self.host_computation_times.push(metrics.host_computation);
        self.total_times.push(metrics.forward_time + metrics.backward_time + metrics.update_time +
                             metrics.memory_transfers + metrics.host_computation);
    }

    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("🚀 Performance Analysis Report\n");
        report.push_str("==============================\n\n");

        // Summary statistics
        if !self.losses.is_empty() {
            let avg_loss = self.losses.iter().sum::<f32>() / self.losses.len() as f32;
            let min_loss = self.losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_loss = self.losses.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            report.push_str(&format!("📊 Loss Statistics:\n"));
            report.push_str(&format!("  Average Loss: {:.6}\n", avg_loss));
            report.push_str(&format!("  Min Loss: {:.6}\n", min_loss));
            report.push_str(&format!("  Max Loss: {:.6}\n", max_loss));
            report.push_str(&format!("  Final Loss: {:.6}\n\n", self.losses.last().unwrap()));
        }

        // Timing breakdown
        if !self.total_times.is_empty() {
            let avg_epoch_time = self.total_times.iter().sum::<Duration>() / self.total_times.len() as u32;
            let total_time: Duration = self.total_times.iter().sum();

            report.push_str(&format!("⏱️  Timing Analysis:\n"));
            report.push_str(&format!("  Total Training Time: {:.2}s\n", total_time.as_secs_f32()));
            report.push_str(&format!("  Average Epoch Time: {:.2}s\n", avg_epoch_time.as_secs_f32()));

            // Detailed breakdown
            if !self.forward_times.is_empty() {
                let avg_forward = self.forward_times.iter().sum::<Duration>() / self.forward_times.len() as u32;
                let avg_backward = self.backward_times.iter().sum::<Duration>() / self.backward_times.len() as u32;
                let avg_update = self.update_times.iter().sum::<Duration>() / self.update_times.len() as u32;
                let avg_memory = self.memory_transfer_times.iter().sum::<Duration>() / self.memory_transfer_times.len() as u32;
                let avg_host = self.host_computation_times.iter().sum::<Duration>() / self.host_computation_times.len() as u32;

                report.push_str(&format!("\n  Average Time Breakdown:\n"));
                report.push_str(&format!("    Forward Pass:  {:.3}ms ({:.1}%)\n",
                    avg_forward.as_millis(),
                    100.0 * avg_forward.as_secs_f32() / avg_epoch_time.as_secs_f32()));
                report.push_str(&format!("    Backward Pass: {:.3}ms ({:.1}%)\n",
                    avg_backward.as_millis(),
                    100.0 * avg_backward.as_secs_f32() / avg_epoch_time.as_secs_f32()));
                report.push_str(&format!("    Weight Update: {:.3}ms ({:.1}%)\n",
                    avg_update.as_millis(),
                    100.0 * avg_update.as_secs_f32() / avg_epoch_time.as_secs_f32()));
                report.push_str(&format!("    Memory Transfer:{:.3}ms ({:.1}%)\n",
                    avg_memory.as_millis(),
                    100.0 * avg_memory.as_secs_f32() / avg_epoch_time.as_secs_f32()));
                report.push_str(&format!("    Host Computation:{:.3}ms ({:.1}%)\n",
                    avg_host.as_millis(),
                    100.0 * avg_host.as_secs_f32() / avg_epoch_time.as_secs_f32()));
            }

            report.push_str("\n");
        }

        // Performance insights
        report.push_str(&format!("💡 Performance Insights:\n"));
        if let Some(&final_loss) = self.losses.last() {
            if final_loss < 0.1 {
                report.push_str("  ✅ Excellent convergence achieved\n");
            } else if final_loss < 1.0 {
                report.push_str("  ⚠️  Good convergence, may need more epochs\n");
            } else {
                report.push_str("  ❌ Poor convergence, check hyperparameters\n");
            }
        }

        // Memory transfer efficiency
        if !self.memory_transfer_times.is_empty() && !self.total_times.is_empty() {
            let avg_memory_pct = self.memory_transfer_times.iter().sum::<Duration>().as_secs_f32() /
                               self.total_times.iter().sum::<Duration>().as_secs_f32() * 100.0;

            if avg_memory_pct > 50.0 {
                report.push_str("  📈 High memory transfer overhead - consider pinned memory\n");
            } else if avg_memory_pct > 20.0 {
                report.push_str("  ⚖️  Moderate memory transfer overhead\n");
            } else {
                report.push_str("  ✅ Low memory transfer overhead - good GPU utilization\n");
            }
        }

        // Metadata
        if !self.metadata.is_empty() {
            report.push_str(&format!("\n🔧 Configuration:\n"));
            for (key, value) in &self.metadata {
                report.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        report
    }

    pub fn save_to_json(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_json(path: &std::path::Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let results = serde_json::from_str(&json)?;
        Ok(results)
    }
}

/// Advanced performance profiler
pub struct PerformanceProfiler {
    start_times: HashMap<String, Instant>,
    measurements: HashMap<String, Vec<Duration>>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            start_times: HashMap::new(),
            measurements: HashMap::new(),
        }
    }

    pub fn start_measurement(&mut self, name: &str) {
        self.start_times.insert(name.to_string(), Instant::now());
    }

    pub fn end_measurement(&mut self, name: &str) {
        if let Some(start) = self.start_times.remove(name) {
            let duration = start.elapsed();
            self.measurements.entry(name.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }

    pub fn get_average(&self, name: &str) -> Option<Duration> {
        self.measurements.get(name)
            .and_then(|measurements| {
                if measurements.is_empty() {
                    None
                } else {
                    Some(measurements.iter().sum::<Duration>() / measurements.len() as u32)
                }
            })
    }

    pub fn get_statistics(&self, name: &str) -> Option<PerformanceStats> {
        self.measurements.get(name)
            .and_then(|measurements| {
                if measurements.is_empty() {
                    None
                } else {
                    let min = *measurements.iter().min().unwrap();
                    let max = *measurements.iter().max().unwrap();
                    let avg = measurements.iter().sum::<Duration>() / measurements.len() as u32;
                    let variance = measurements.iter()
                        .map(|&d| {
                            let diff = d.as_secs_f32() - avg.as_secs_f32();
                            diff * diff
                        })
                        .sum::<f32>() / measurements.len() as f32;
                    let std_dev = variance.sqrt();

                    Some(PerformanceStats {
                        count: measurements.len(),
                        min,
                        max,
                        average: avg,
                        std_dev: Duration::from_secs_f32(std_dev),
                    })
                }
            })
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("🔬 Detailed Performance Profile\n");
        report.push_str("================================\n\n");

        for (name, measurements) in &self.measurements {
            if let Some(stats) = self.get_statistics(name) {
                report.push_str(&format!("📍 {}:\n", name));
                report.push_str(&format!("  Samples: {}\n", stats.count));
                report.push_str(&format!("  Average: {:.3}ms\n", stats.average.as_millis()));
                report.push_str(&format!("  Min:     {:.3}ms\n", stats.min.as_millis()));
                report.push_str(&format!("  Max:     {:.3}ms\n", stats.max.as_millis()));
                report.push_str(&format!("  StdDev:  {:.3}ms\n\n", stats.std_dev.as_millis()));
            }
        }

        report
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub count: usize,
    pub min: Duration,
    pub max: Duration,
    pub average: Duration,
    pub std_dev: Duration,
}
