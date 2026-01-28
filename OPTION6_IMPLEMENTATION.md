# Option 6 Implementation: Address-Based Neighbor Probing

**Implementation Date:** 2026-01-28  
**Status:** ✓ COMPLETE

## Summary

This implementation delivers mandatory address-based neighbor probing (Option 6) with a full 64-slot neighbor structure, integrating with the existing geometric attention mechanism. The implementation provides O(N × 64 × d') complexity attention without dense matrix multiplication.

## Key Features

### 1. 64-Slot Neighbor Structure (Mandatory, No Fallbacks)
- **32 nearest neighbors**: Highest similarity (similarity grounding)
- **16 attractors**: Top similarity after nearest (reinforcing evidence)
- **16 repulsors**: Lowest similarity (contrastive evidence)
- Selection via `select_neighbors()` method with role-based partitioning
- Automatic repetition if fewer than 64 candidates available

### 2. 6 Similarity Scores Per Neighbor (Mandatory)
- **Score 0**: Cosine similarity (geometric baseline)
- **Scores 1-5**: Learned similarity metrics via projection
- Always computed internally if not provided externally
- No fallback to empty slots—all 6 scores populated

### 3. Per-Neighbor Geometric Features
Each of the 64 neighbor blocks contains:
- **Value** (64 dims): Interaction output vector
- **Neighbor metric** (16 dims): Metric features of this neighbor
- **Neighbor transport** (16 dims): Transport features of this neighbor
- **Scores** (6 dims): 6 similarity types
- **Coords** (16 dims): Routing information
- **Total per neighbor**: 118 floats

### 4. Collision Avoidance System
- 64-bit route hash via learned projection (`route_hash_proj`)
- First 32 bits stored in ECC field for uniqueness
- Helper functions:
  - `check_address_collisions()`: Detects collision pairs
  - `compute_address_uniqueness_score()`: Returns uniqueness metric [0,1]
- Provides address-space entropy for collision mitigation

### 5. Integration with GeometricAttention
- New method: `probe_address_neighbors()` consumes full Address structure
- Updated `forward()` accepts `Q_address` parameter
- Attention weighting using 6 similarity scores with role-typed boosting
- Dedicated `neighbor_value_proj` layer (d'=64 → d_model)
- Proper normalization: weights renormalized after Born×Gibbs×Softmax

## Implementation Details

### Total Address Dimension
- **d=512 (default)**: 9122 floats total
  - Core: 512
  - Metric: 512
  - Transport: 512
  - Neighbors: 64 × 118 = 7552
  - ECC: 32
  - Timestamps: 2

### Files Modified
1. **inference/address.py**
   - Enhanced `AddressConfig` with 6 similarity scores (m=6)
   - Added per-neighbor metric/transport dimensions
   - Implemented `compute_similarity_scores()` for 6-score computation
   - Implemented `select_neighbors()` for role-typed selection
   - Added collision-avoidance helpers
   - Complete docstring and schema documentation

2. **inference/geometric_attention.py**
   - Added Address import
   - Implemented `probe_address_neighbors()` method
   - Updated `forward()` to support Address-based routing
   - Added `neighbor_value_proj` layer
   - Fixed attention weight normalization

3. **WIRING_PLAN.md**
   - Updated with full Option 6 implementation details
   - Documented 64-slot structure, 6 scores, and collision avoidance
   - Added implementation summary section

4. **kernels/metric_context.py**
   - Fixed exception handling in context manager

### Configuration Flags
- `AddressConfig.enable_address_probing`: Default `True` for Option 6
- When enabled: mandatory 64-slot probing active
- When disabled: legacy behavior (testing only)

## Testing

### Unit Tests (tests/test_address_builder.py)
Comprehensive pytest suite covering:
1. ✓ Config dimensions validation
2. ✓ Address shape correctness
3. ✓ 64 neighbors populated
4. ✓ 6 score channels per neighbor
5. ✓ Role-typed partitions (32+16+16)
6. ✓ Metric/transport per neighbor
7. ✓ ECC and timestamps present
8. ✓ Collision checking
9. ✓ Uniqueness score computation
10. ✓ Individual neighbor access
11. ✓ Fallback with few neighbors
12. ✓ Self-similarity fallback
13. ✓ Address-probing flag

### Standalone Validation
- All 10 test categories passing
- Integration with attention mechanism validated
- Multi-token scenarios verified

## Code Quality

### Code Review
- ✓ All review comments addressed
- ✓ Documentation inconsistencies fixed
- ✓ Dead code removed
- ✓ Normalization issues resolved
- ✓ Proper projections added
- ✓ Exception handling fixed

### Security
- ✓ CodeQL scan: 0 vulnerabilities
- ✓ No security issues detected
- ✓ Safe tensor operations throughout

## ECC and Timestamps

### Behavior
- **Present**: Always included in address structure
- **Excluded from scoring**: Not used in neighbor similarity computation
- **Purpose**:
  - ECC: Collision-avoidance hash for uniqueness
  - Timestamps: Temporal ordering (internal_time, wall_time)

### Optionality Definition
- "Optional" means: only during controlled maintenance (temporary disable)
- At runtime with `enable_address_probing=True`: ALL fields mandatory
- No opt-out for neighbor slots, scores, or geometric features

## Integration Path

The implementation provides three input modes for `GeometricAttention.forward()`:

1. **Address-based (Option 6 Extended, preferred)**:
   ```python
   output, weights = attention(Q_address=address)
   ```
   Uses full Address structure with 64 neighbors and 6 scores

2. **Neighbor embeddings (Option 6)**:
   ```python
   output, weights = attention(Q_input=Q, neighbor_embeddings=neighbors)
   ```
   Uses raw neighbor embeddings without Address wrapper

3. **Legacy K/V**:
   ```python
   output, weights = attention(Q_input=Q, K=K, V=V)
   ```
   Standard O(N²) attention (backward compatibility)

## Next Steps

To fully utilize Option 6 in production:

1. **Wire AddressBuilder into data pipeline**:
   - Build addresses from token embeddings at input
   - Populate neighbors from context or retrieval

2. **Replace K/V matmul with Address routing**:
   - Use Address-based path in main attention layers
   - Disable legacy K/V path in production

3. **Tune hyperparameters**:
   - Adjust role weights (attractors=1.5, repulsors=-0.5)
   - Tune temperature for Born×Gibbs×Softmax
   - Experiment with learned vs. cosine similarity weights

4. **Monitor metrics**:
   - Track collision rates via `check_address_collisions()`
   - Monitor uniqueness via `compute_address_uniqueness_score()`
   - Validate attention weight distributions

## References

- **WIRING_PLAN.md**: Full architectural documentation
- **inference/address.py**: Core implementation
- **inference/geometric_attention.py**: Integration layer
- **tests/test_address_builder.py**: Comprehensive test suite

---

**Implementation Complete** ✓  
All requirements met, tested, documented, and ready for integration.
