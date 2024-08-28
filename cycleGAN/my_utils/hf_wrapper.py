hf_helpers_available = False
try:
    import hf_helpers
    print("HFHelpers available")
    hf_helpers_available = True
except ImportError:
    pass

if hf_helpers_available:
    from hf_helpers.debugging import debug_on_error
else:
    debug_on_error = lambda x: x
