use std::fs::File;
use std::io::Read;
use std::path::Path;
use anyhow::{Result, anyhow};

#[derive(Debug, Clone)]
pub struct MnistData {
    pub train_images: Vec<f32>,
    pub train_labels: Vec<f32>,
    pub test_images: Vec<f32>,
    pub test_labels: Vec<f32>,
    pub num_train_samples: usize,
    pub num_test_samples: usize,
}

impl MnistData {
    /// Load MNIST data from binary files (compatible with existing format)
    pub fn load_from_binary(data_dir: &Path, max_train_samples: usize) -> Result<Self> {
        let train_images_path = data_dir.join("X_train.bin");
        let train_labels_path = data_dir.join("y_train.bin");
        let test_images_path = data_dir.join("X_test.bin");
        let test_labels_path = data_dir.join("y_test.bin");

        // Load training data
        let mut train_images = load_binary_file(&train_images_path)?;
        let mut train_labels = load_binary_file(&train_labels_path)?;

        // Apply MNIST normalization (mean=0.1307, std=0.3081)
        normalize_mnist_data(&mut train_images);

        // Limit training samples if specified
        let actual_train_samples = (max_train_samples * 784).min(train_images.len()) / 784;
        train_images.truncate(actual_train_samples * 784);
        train_labels.truncate(actual_train_samples);

        // Load test data
        let mut test_images = load_binary_file(&test_images_path)?;
        let test_labels = load_binary_file(&test_labels_path)?;
        normalize_mnist_data(&mut test_images);

        let num_test_samples = test_images.len() / 784;

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            num_train_samples: actual_train_samples,
            num_test_samples,
        })
    }

    /// Get batch of training data
    pub fn get_train_batch(&self, batch_idx: usize, batch_size: usize) -> Option<(&[f32], &[f32])> {
        let start_idx = batch_idx * batch_size;
        let end_idx = (start_idx + batch_size).min(self.num_train_samples);

        if start_idx >= self.num_train_samples {
            return None;
        }

        let images = &self.train_images[start_idx * 784..end_idx * 784];
        let labels = &self.train_labels[start_idx..end_idx];

        Some((images, labels))
    }

    /// Shuffle training data
    pub fn shuffle_train_data(&mut self) {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..self.num_train_samples).collect();
        indices.shuffle(&mut rng);

        // Reorder images and labels
        let mut new_images = vec![0.0f32; self.train_images.len()];
        let mut new_labels = vec![0.0f32; self.train_labels.len()];

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            let old_img_start = old_idx * 784;
            let new_img_start = new_idx * 784;

            new_images[new_img_start..new_img_start + 784]
                .copy_from_slice(&self.train_images[old_img_start..old_img_start + 784]);
            new_labels[new_idx] = self.train_labels[old_idx];
        }

        self.train_images = new_images;
        self.train_labels = new_labels;
    }
}

/// Load binary file as f32 vector
fn load_binary_file(path: &Path) -> Result<Vec<f32>> {
    let mut file = File::open(path)
        .map_err(|e| anyhow!("Failed to open file {:?}: {}", path, e))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| anyhow!("Failed to read file {:?}: {}", path, e))?;

    // Convert bytes to f32 (assuming little-endian)
    if buffer.len() % 4 != 0 {
        return Err(anyhow!("File size not divisible by 4: {}", buffer.len()));
    }

    let mut data = Vec::with_capacity(buffer.len() / 4);
    for chunk in buffer.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        data.push(value);
    }

    Ok(data)
}

/// Apply MNIST normalization: (x - mean) / std
fn normalize_mnist_data(data: &mut [f32]) {
    const MEAN: f32 = 0.1307;
    const STD: f32 = 0.3081;

    for value in data.iter_mut() {
        *value = (*value - MEAN) / STD;
    }
}
