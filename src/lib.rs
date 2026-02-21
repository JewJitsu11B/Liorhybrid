//! A simple mathematical utilities library
//!
//! This library provides common mathematical operations and utilities.

/// Calculates the factorial of a non-negative integer.
///
/// # Arguments
///
/// * `n` - A non-negative integer
///
/// # Returns
///
/// The factorial of `n`
///
/// # Panics
///
/// Panics if the result would overflow u64
///
/// # Examples
///
/// ```
/// use rust_project::factorial;
///
/// assert_eq!(factorial(5), 120);
/// assert_eq!(factorial(0), 1);
/// ```
pub fn factorial(n: u64) -> u64 {
    match n {
        0 | 1 => 1,
        _ => (2..=n).product(),
    }
}

/// Checks if a number is prime.
///
/// # Arguments
///
/// * `n` - The number to check
///
/// # Returns
///
/// `true` if the number is prime, `false` otherwise
///
/// # Examples
///
/// ```
/// use rust_project::is_prime;
///
/// assert!(is_prime(7));
/// assert!(!is_prime(4));
/// ```
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    let sqrt_n = (n as f64).sqrt() as u64;
    for i in (3..=sqrt_n).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

/// Calculates the greatest common divisor of two numbers using Euclid's algorithm.
///
/// # Arguments
///
/// * `a` - First number
/// * `b` - Second number
///
/// # Returns
///
/// The GCD of `a` and `b`
///
/// # Examples
///
/// ```
/// use rust_project::gcd;
///
/// assert_eq!(gcd(48, 18), 6);
/// assert_eq!(gcd(17, 13), 1);
/// ```
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Calculates the least common multiple of two numbers.
///
/// # Arguments
///
/// * `a` - First number
/// * `b` - Second number
///
/// # Returns
///
/// The LCM of `a` and `b`
///
/// # Examples
///
/// ```
/// use rust_project::lcm;
///
/// assert_eq!(lcm(12, 18), 36);
/// assert_eq!(lcm(5, 7), 35);
/// ```
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a * b) / gcd(a, b)
}

/// Calculates the nth Fibonacci number.
///
/// # Arguments
///
/// * `n` - The position in the Fibonacci sequence
///
/// # Returns
///
/// The nth Fibonacci number
///
/// # Examples
///
/// ```
/// use rust_project::fibonacci;
///
/// assert_eq!(fibonacci(0), 0);
/// assert_eq!(fibonacci(1), 1);
/// assert_eq!(fibonacci(10), 55);
/// ```
pub fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let mut a = 0;
            let mut b = 1;
            for _ in 2..=n {
                let temp = a + b;
                a = b;
                b = temp;
            }
            b
        }
    }
}

/// A simple calculator for basic arithmetic operations.
pub struct Calculator {
    memory: f64,
}

impl Calculator {
    /// Creates a new Calculator with memory initialized to 0.
    pub fn new() -> Self {
        Calculator { memory: 0.0 }
    }

    /// Adds two numbers.
    pub fn add(&self, a: f64, b: f64) -> f64 {
        a + b
    }

    /// Subtracts b from a.
    pub fn subtract(&self, a: f64, b: f64) -> f64 {
        a - b
    }

    /// Multiplies two numbers.
    pub fn multiply(&self, a: f64, b: f64) -> f64 {
        a * b
    }

    /// Divides a by b.
    ///
    /// # Returns
    ///
    /// `Some(result)` if b is not zero, `None` otherwise
    pub fn divide(&self, a: f64, b: f64) -> Option<f64> {
        if b == 0.0 {
            None
        } else {
            Some(a / b)
        }
    }

    /// Stores a value in memory.
    pub fn store(&mut self, value: f64) {
        self.memory = value;
    }

    /// Recalls the value from memory.
    pub fn recall(&self) -> f64 {
        self.memory
    }

    /// Clears the memory.
    pub fn clear(&mut self) {
        self.memory = 0.0;
    }
}

impl Default for Calculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Factorial tests
    #[test]
    fn test_factorial_zero() {
        assert_eq!(factorial(0), 1);
    }

    #[test]
    fn test_factorial_one() {
        assert_eq!(factorial(1), 1);
    }

    #[test]
    fn test_factorial_small_numbers() {
        assert_eq!(factorial(2), 2);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(4), 24);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_factorial_larger_number() {
        assert_eq!(factorial(10), 3_628_800);
    }

    #[test]
    fn test_factorial_edge_case() {
        assert_eq!(factorial(20), 2_432_902_008_176_640_000);
    }

    // Prime number tests
    #[test]
    fn test_is_prime_small_primes() {
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(is_prime(11));
    }

    #[test]
    fn test_is_prime_composites() {
        assert!(!is_prime(4));
        assert!(!is_prime(6));
        assert!(!is_prime(8));
        assert!(!is_prime(9));
        assert!(!is_prime(10));
    }

    #[test]
    fn test_is_prime_edge_cases() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
    }

    #[test]
    fn test_is_prime_larger_primes() {
        assert!(is_prime(97));
        assert!(is_prime(101));
        assert!(!is_prime(100));
    }

    #[test]
    fn test_is_prime_large_composite() {
        assert!(!is_prime(1000));
    }

    // GCD tests
    #[test]
    fn test_gcd_basic() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(18, 48), 6);
    }

    #[test]
    fn test_gcd_coprime() {
        assert_eq!(gcd(17, 13), 1);
    }

    #[test]
    fn test_gcd_same_numbers() {
        assert_eq!(gcd(42, 42), 42);
    }

    #[test]
    fn test_gcd_with_zero() {
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
    }

    #[test]
    fn test_gcd_one() {
        assert_eq!(gcd(1, 1), 1);
        assert_eq!(gcd(1, 100), 1);
    }

    // LCM tests
    #[test]
    fn test_lcm_basic() {
        assert_eq!(lcm(12, 18), 36);
    }

    #[test]
    fn test_lcm_coprime() {
        assert_eq!(lcm(5, 7), 35);
    }

    #[test]
    fn test_lcm_with_zero() {
        assert_eq!(lcm(0, 5), 0);
        assert_eq!(lcm(5, 0), 0);
    }

    #[test]
    fn test_lcm_same_numbers() {
        assert_eq!(lcm(7, 7), 7);
    }

    #[test]
    fn test_lcm_one() {
        assert_eq!(lcm(1, 5), 5);
    }

    // Fibonacci tests
    #[test]
    fn test_fibonacci_base_cases() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
    }

    #[test]
    fn test_fibonacci_sequence() {
        assert_eq!(fibonacci(2), 1);
        assert_eq!(fibonacci(3), 2);
        assert_eq!(fibonacci(4), 3);
        assert_eq!(fibonacci(5), 5);
        assert_eq!(fibonacci(6), 8);
    }

    #[test]
    fn test_fibonacci_larger() {
        assert_eq!(fibonacci(10), 55);
        assert_eq!(fibonacci(20), 6765);
    }

    // Calculator tests
    #[test]
    fn test_calculator_new() {
        let calc = Calculator::new();
        assert_eq!(calc.recall(), 0.0);
    }

    #[test]
    fn test_calculator_default() {
        let calc = Calculator::default();
        assert_eq!(calc.recall(), 0.0);
    }

    #[test]
    fn test_calculator_add() {
        let calc = Calculator::new();
        assert_eq!(calc.add(5.0, 3.0), 8.0);
        assert_eq!(calc.add(-5.0, 3.0), -2.0);
        assert_eq!(calc.add(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_calculator_add_floats() {
        let calc = Calculator::new();
        let result = calc.add(0.1, 0.2);
        assert!((result - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_calculator_subtract() {
        let calc = Calculator::new();
        assert_eq!(calc.subtract(10.0, 3.0), 7.0);
        assert_eq!(calc.subtract(3.0, 10.0), -7.0);
        assert_eq!(calc.subtract(5.0, 5.0), 0.0);
    }

    #[test]
    fn test_calculator_multiply() {
        let calc = Calculator::new();
        assert_eq!(calc.multiply(4.0, 5.0), 20.0);
        assert_eq!(calc.multiply(-3.0, 4.0), -12.0);
        assert_eq!(calc.multiply(0.0, 100.0), 0.0);
    }

    #[test]
    fn test_calculator_divide() {
        let calc = Calculator::new();
        assert_eq!(calc.divide(10.0, 2.0), Some(5.0));
        assert_eq!(calc.divide(7.0, 2.0), Some(3.5));
    }

    #[test]
    fn test_calculator_divide_by_zero() {
        let calc = Calculator::new();
        assert_eq!(calc.divide(10.0, 0.0), None);
    }

    #[test]
    fn test_calculator_memory() {
        let mut calc = Calculator::new();
        calc.store(42.0);
        assert_eq!(calc.recall(), 42.0);
        calc.store(100.0);
        assert_eq!(calc.recall(), 100.0);
    }

    #[test]
    fn test_calculator_clear() {
        let mut calc = Calculator::new();
        calc.store(42.0);
        assert_eq!(calc.recall(), 42.0);
        calc.clear();
        assert_eq!(calc.recall(), 0.0);
    }

    #[test]
    fn test_calculator_negative_memory() {
        let mut calc = Calculator::new();
        calc.store(-15.5);
        assert_eq!(calc.recall(), -15.5);
    }

    // Integration-style tests
    #[test]
    fn test_factorial_and_prime() {
        // Factorial of 5 is 120, which is not prime
        assert_eq!(factorial(5), 120);
        assert!(!is_prime(factorial(5)));

        // Factorial of 2 is 2, which is prime
        assert_eq!(factorial(2), 2);
        assert!(is_prime(factorial(2)));
    }

    #[test]
    fn test_gcd_lcm_relationship() {
        let a = 12;
        let b = 18;
        // GCD * LCM = a * b
        assert_eq!(gcd(a, b) * lcm(a, b), a * b);
    }

    #[test]
    fn test_calculator_chain_operations() {
        let mut calc = Calculator::new();
        let result1 = calc.add(10.0, 5.0);
        calc.store(result1);
        let result2 = calc.multiply(calc.recall(), 2.0);
        assert_eq!(result2, 30.0);
    }

    // Edge case and regression tests
    #[test]
    fn test_fibonacci_does_not_overflow_early() {
        // Just verify it completes without panic for reasonable values
        let _ = fibonacci(50);
    }

    #[test]
    fn test_is_prime_even_number_optimization() {
        // Verify that even numbers > 2 are correctly identified as not prime
        for n in (4..100).step_by(2) {
            assert!(!is_prime(n), "Failed for {}", n);
        }
    }

    #[test]
    fn test_gcd_commutative() {
        assert_eq!(gcd(24, 36), gcd(36, 24));
    }

    #[test]
    fn test_calculator_divide_negative_numbers() {
        let calc = Calculator::new();
        assert_eq!(calc.divide(-10.0, 2.0), Some(-5.0));
        assert_eq!(calc.divide(10.0, -2.0), Some(-5.0));
        assert_eq!(calc.divide(-10.0, -2.0), Some(5.0));
    }
}