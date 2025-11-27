use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;

mod neural_net;
mod data_loader;
mod benchmarks;

use neural_net::{NeuralNetwork, Backend, TrainingConfig};
use data_loader::MnistData;
use benchmarks::BenchmarkResults;

#[derive(Parser)]
#[command(name = "MNIST CUDA Rust")]
#[command(about = "Advanced MNIST neural network training with multiple backends")]
struct Args {
    /// Backend to use for computation
    #[arg(short, long, default_value = "cuda-optimized")]
    backend: BackendType,

    /// Batch size for training
    #[arg(short, long, default_value = "8")]
    batch_size: usize,

    /// Number of epochs
    #[arg(short = 'e', long, default_value = "10")]
    epochs: usize,

    /// Learning rate
    #[arg(short = 'r', long, default_value = "0.01")]
    learning_rate: f32,

    /// Training samples to use
    #[arg(short = 's', long, default_value = "10000")]
    train_samples: usize,

    /// Enable mixed precision training
    #[arg(long)]
    mixed_precision: bool,

    /// Enable async operations
    #[arg(long)]
    async_ops: bool,

    /// Data directory
    #[arg(short = 'd', long, default_value = "data")]
    data_dir: PathBuf,

    /// Enable benchmarking mode
    #[arg(long)]
    benchmark: bool,

    /// Enable correctness verification
    #[arg(long)]
    verify: bool,
}

#[derive(Clone, ValueEnum)]
enum BackendType {
    Cpu,
    CudaNaive,
    CudaOptimized,
    CudaAsync,
    MixedPrecision,
}

impl From<BackendType> for Backend {
    fn from(bt: BackendType) -> Self {
        match bt {
            BackendType::Cpu => Backend::Cpu,
            BackendType::CudaNaive => Backend::CudaNaive,
            BackendType::CudaOptimized => Backend::CudaOptimized,
            BackendType::CudaAsync => Backend::CudaAsync,
            BackendType::MixedPrecision => Backend::CudaMixedPrecision,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("🧠 Advanced MNIST Neural Network Training");
    println!("==========================================");
    println!("Backend: {:?}", args.backend);
    println!("Batch Size: {}", args.batch_size);
    println!("Epochs: {}", args.epochs);
    println!("Learning Rate: {}", args.learning_rate);
    println!("Mixed Precision: {}", args.mixed_precision);
    println!("Async Operations: {}", args.async_ops);
    println!();

    // Load MNIST data
    let data_start = Instant::now();
    let data = MnistData::load_from_binary(&args.data_dir, args.train_samples)?;
    println!("📊 Data loaded in {:.2}ms", data_start.elapsed().as_millis());

    // Create neural network
    let config = TrainingConfig {
        input_size: 784,
        hidden_size: 256,
        output_size: 10,
        batch_size: args.batch_size,
        epochs: args.epochs,
        learning_rate: args.learning_rate,
        mixed_precision: args.mixed_precision,
        async_operations: args.async_ops,
    };

    let mut network = NeuralNetwork::new(args.backend.into(), config).await?;

    // Training loop with progress tracking
    let train_start = Instant::now();
    let mut benchmark_results = BenchmarkResults::new();

    println!("\n🚀 Starting training...");
    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();

        let (loss, epoch_metrics) = network.train_epoch(&data).await?;
        benchmark_results.record_epoch(epoch, loss, epoch_metrics);

        let epoch_time = epoch_start.elapsed();
        println!("Epoch {:2}/{:2} | Loss: {:.6} | Time: {:.2}s",
                epoch + 1, args.epochs, loss, epoch_time.as_secs_f32());
    }

    let total_train_time = train_start.elapsed();
    println!("\n✅ Training completed in {:.2}s", total_train_time.as_secs_f32());

    // Benchmarking
    if args.benchmark {
        println!("\n📈 Performance Analysis:");
        println!("{}", benchmark_results.generate_report());
    }

    // Verification
    if args.verify {
        println!("\n🔍 Running correctness verification...");
        let accuracy = network.evaluate(&data)?;
        println!("Test Accuracy: {:.2}%", accuracy * 100.0);
    }

    // Final statistics
    println!("\n🎯 Final Results:");
    println!("Total Training Time: {:.2}s", total_train_time.as_secs_f32());
    println!("Average Epoch Time: {:.2}s", total_train_time.as_secs_f32() / args.epochs as f32);
    println!("Samples/Second: {:.0}", args.train_samples as f32 * args.epochs as f32 / total_train_time.as_secs_f32());

    Ok(())
}
