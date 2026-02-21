//! Integration tests for the rust-project library
//!
//! These tests verify that the library functions work correctly together
//! and can be used as intended by external crates.

use rust_project::*;

#[test]
fn test_mathematical_operations_integration() {
    // Test that mathematical functions work together
    let n = 5;
    let fact = factorial(n);
    assert_eq!(fact, 120);

    // Check if factorial result is prime
    assert!(!is_prime(fact));

    // Calculate GCD and LCM of two factorials
    let fact3 = factorial(3);
    let fact4 = factorial(4);
    assert_eq!(gcd(fact3, fact4), 6);
    assert_eq!(lcm(fact3, fact4), 24);
}

#[test]
fn test_fibonacci_prime_analysis() {
    // Find Fibonacci numbers that are prime
    let fib_numbers = vec![
        (2, fibonacci(2)),
        (3, fibonacci(3)),
        (4, fibonacci(4)),
        (5, fibonacci(5)),
        (7, fibonacci(7)),
    ];

    // Fibonacci(3) = 2 is prime
    assert!(is_prime(fib_numbers[1].1));

    // Fibonacci(4) = 3 is prime
    assert!(is_prime(fib_numbers[2].1));

    // Fibonacci(5) = 5 is prime
    assert!(is_prime(fib_numbers[3].1));

    // Fibonacci(7) = 13 is prime
    assert!(is_prime(fib_numbers[4].1));
}

#[test]
fn test_calculator_with_factorial() {
    let mut calc = Calculator::new();

    // Calculate factorial of small numbers and perform operations
    let fact3 = factorial(3);
    let fact4 = factorial(4);

    let sum = calc.add(fact3 as f64, fact4 as f64);
    assert_eq!(sum, 30.0); // 6 + 24 = 30

    calc.store(sum);
    let result = calc.divide(calc.recall(), 5.0);
    assert_eq!(result, Some(6.0));
}

#[test]
fn test_calculator_complex_workflow() {
    let mut calc = Calculator::new();

    // Simulate a series of calculations
    let step1 = calc.add(100.0, 50.0); // 150
    calc.store(step1);

    let step2 = calc.multiply(calc.recall(), 0.1); // 15
    calc.store(step2);

    let step3 = calc.subtract(calc.recall(), 5.0); // 10
    calc.store(step3);

    let step4 = calc.divide(calc.recall(), 2.0).unwrap(); // 5
    calc.store(step4);

    assert_eq!(calc.recall(), 5.0);

    calc.clear();
    assert_eq!(calc.recall(), 0.0);
}

#[test]
fn test_gcd_with_fibonacci() {
    // Test GCD property with Fibonacci numbers
    // GCD(F(n), F(n+1)) = 1 for consecutive Fibonacci numbers
    for i in 1..10 {
        let fib_n = fibonacci(i);
        let fib_n_plus_1 = fibonacci(i + 1);
        assert_eq!(gcd(fib_n, fib_n_plus_1), 1);
    }
}

#[test]
fn test_prime_factorization_analysis() {
    // Test prime-related properties
    let primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];

    for &p in &primes {
        assert!(is_prime(p));

        // GCD of any two distinct primes should be 1
        for &q in &primes {
            if p != q {
                assert_eq!(gcd(p, q), 1);
            }
        }
    }
}

#[test]
fn test_lcm_with_prime_numbers() {
    // LCM of two primes is their product
    let p1 = 7;
    let p2 = 11;

    assert!(is_prime(p1));
    assert!(is_prime(p2));
    assert_eq!(lcm(p1, p2), p1 * p2);
}

#[test]
fn test_calculator_precision() {
    let calc = Calculator::new();

    // Test division precision
    let result = calc.divide(1.0, 3.0).unwrap();
    let triple = calc.multiply(result, 3.0);

    // Should be close to 1.0 within floating point precision
    assert!((triple - 1.0).abs() < 1e-10);
}

#[test]
fn test_factorial_sum_pattern() {
    // Test mathematical properties
    // Sum of factorials: 1! + 2! + 3! = 1 + 2 + 6 = 9
    let sum = factorial(1) + factorial(2) + factorial(3);
    assert_eq!(sum, 9);

    let mut calc = Calculator::new();
    calc.store(sum as f64);
    assert_eq!(calc.recall(), 9.0);
}

#[test]
fn test_error_handling_in_workflow() {
    let calc = Calculator::new();

    // Test that division by zero is handled properly
    let result = calc.divide(100.0, 0.0);
    assert!(result.is_none());

    // Verify we can continue after error
    let valid_result = calc.divide(100.0, 10.0);
    assert_eq!(valid_result, Some(10.0));
}

#[test]
fn test_large_number_operations() {
    // Test with larger numbers
    let large1 = factorial(15);
    let large2 = factorial(16);

    let g = gcd(large1, large2);
    let l = lcm(large1, large2);

    // Verify GCD * LCM = a * b
    assert_eq!(g * l, large1 * large2);
}

#[test]
fn test_calculator_memory_persistence() {
    let mut calc = Calculator::new();

    // Store a value
    calc.store(123.456);

    // Perform operations that don't affect memory
    let _ = calc.add(1.0, 2.0);
    let _ = calc.multiply(3.0, 4.0);

    // Memory should still have the stored value
    assert_eq!(calc.recall(), 123.456);
}

#[test]
fn test_fibonacci_growth_rate() {
    // Verify that Fibonacci numbers grow as expected
    for i in 2..20 {
        let fib_prev = fibonacci(i - 1);
        let fib_curr = fibonacci(i);
        let fib_next = fibonacci(i + 1);

        // Fibonacci property: F(n) = F(n-1) + F(n-2)
        assert_eq!(fib_next, fib_curr + fib_prev);
    }
}

#[test]
fn test_multiple_calculator_instances() {
    let mut calc1 = Calculator::new();
    let mut calc2 = Calculator::new();

    calc1.store(100.0);
    calc2.store(200.0);

    // Verify they maintain separate memory
    assert_eq!(calc1.recall(), 100.0);
    assert_eq!(calc2.recall(), 200.0);

    calc1.clear();
    assert_eq!(calc1.recall(), 0.0);
    assert_eq!(calc2.recall(), 200.0); // calc2 should be unaffected
}

#[test]
fn test_edge_cases_comprehensive() {
    // Test various edge cases together
    assert_eq!(factorial(0), 1);
    assert_eq!(fibonacci(0), 0);
    assert!(!is_prime(0));
    assert!(!is_prime(1));
    assert_eq!(gcd(0, 5), 5);
    assert_eq!(lcm(0, 5), 0);
}

#[test]
fn test_calculator_with_negative_numbers() {
    let calc = Calculator::new();

    // Test all operations with negative numbers
    assert_eq!(calc.add(-5.0, -3.0), -8.0);
    assert_eq!(calc.subtract(-10.0, -3.0), -7.0);
    assert_eq!(calc.multiply(-4.0, -5.0), 20.0);
    assert_eq!(calc.divide(-20.0, -4.0), Some(5.0));
}

#[test]
fn test_is_prime_performance() {
    // Test that is_prime handles reasonably large numbers
    // without taking too long (should complete quickly)
    assert!(is_prime(7919)); // A known prime
    assert!(!is_prime(7920)); // Not prime
}

#[test]
fn test_gcd_lcm_properties() {
    // Test mathematical properties that should always hold
    let a = 24;
    let b = 36;
    let c = 18;

    // GCD is commutative
    assert_eq!(gcd(a, b), gcd(b, a));

    // LCM is commutative
    assert_eq!(lcm(a, b), lcm(b, a));

    // GCD is associative
    assert_eq!(gcd(gcd(a, b), c), gcd(a, gcd(b, c)));

    // LCM relationship: lcm(a, b) * gcd(a, b) = a * b
    assert_eq!(lcm(a, b) * gcd(a, b), a * b);
}

#[test]
fn test_default_trait_implementation() {
    // Verify Default trait works correctly
    let calc1 = Calculator::default();
    let calc2 = Calculator::new();

    assert_eq!(calc1.recall(), calc2.recall());
}