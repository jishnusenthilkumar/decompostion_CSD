import numpy as np
from csd_to_native_gates import csd_to_native_pipeline, print_results
from scipy.stats import unitary_group



def test_random_unitary():
    """Test with random unitary matrix."""
    from scipy.stats import unitary_group
    n = 1 # Change n for watching different sizes
    U = unitary_group.rvs(2**n)
    
    
    print("\nTesting random 4×4 unitary matrix:")
    
    results = csd_to_native_pipeline(U, optimize=True)
    print_results(results)
    
    return results



if __name__ == "__main__":
    print("Native Gate Decomposition Pipeline Test")
    print("="*50)
    
   
    
    test_random_unitary()
   
    
    print("\n✅ All tests completed!")
