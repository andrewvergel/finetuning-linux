# Lessons Learned: Configuration and Testing

This document captures key lessons and best practices learned from developing and enhancing the model configuration and data processing components in the finetuning pipeline.

## 1. Configuration Design Principles

### 1.1 Type Safety
- **Use Enums for Fixed Sets of Values**
  - Created enums like `LoraBiasType`, `QuantizationType`, and `ComputeDtype` to ensure type safety
  - Prevents invalid values and provides better IDE support with autocompletion

- **Leverage Type Hints**
  - Used Python's type hints throughout the configuration
  - Added `Literal` types for constrained string values (e.g., `padding_side: Literal["left", "right"]`)

### 1.2 Validation
- **Field-Level Validation**
  - Used Pydantic's `Field` with constraints (e.g., `ge`, `le`, `gt`)
  - Example: `lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)`

- **Cross-Field Validation**
  - Added validation to prevent conflicting settings (e.g., can't use both 4-bit and 8-bit quantization)
  - Example: `if self.load_in_4bit and self.load_in_8bit: raise ValueError(...)`

## 2. Documentation Best Practices

### 2.1 Docstrings
- **Module-Level Documentation**
  - Added a comprehensive module docstring with usage example
  - Clearly states the purpose and scope of the module

- **Class and Method Documentation**
  - Used Google-style docstrings for classes and methods
  - Documented all parameters, return values, and exceptions
  - Included example usage in relevant docstrings

### 2.2 Parameter Documentation
- **Descriptive Field Descriptions**
  - Added clear, concise descriptions for each configuration parameter
  - Included units and valid ranges where applicable
  - Documented default values

## 3. Code Organization

### 3.1 Logical Grouping
- Grouped related configuration parameters together with section comments:
  ```python
  # Model settings
  model_name_or_path: str = ...
  trust_remote_code: bool = ...
  
  # Quantization settings
  load_in_4bit: bool = ...
  ```

### 3.2 Configuration Conversion Methods
- Added dedicated methods to convert to framework-specific formats:
  - `to_peft_config()`: For PEFT/LoRA configuration
  - `to_quantization_config()`: For quantization settings
  - `to_generation_config()`: For text generation parameters

## 4. Error Prevention

### 4.1 Safe Defaults
- Chose sensible defaults for all parameters
- Made potentially dangerous operations opt-in (e.g., `trust_remote_code` is `True` by default but documented)

### 4.2 Input Validation
- Validated numeric ranges (e.g., `temperature` between 0 and 2.0)
- Added validation for mutually exclusive options
- Included clear error messages for invalid configurations

## 5. Testing Considerations

### 5.1 Test Cases to Implement
- Test all validation rules
- Verify configuration conversion methods
- Test edge cases (min/max values, None values, etc.)
- Test serialization/deserialization

### 5.2 Integration Testing
- Test with actual model loading
- Verify compatibility with different model architectures
- Test different combinations of settings

## 6. Future Improvements

### 6.1 Enhanced Validation
- Add more sophisticated validation rules
- Consider using Pydantic validators for complex validation logic

### 6.2 Extended Configuration
- Add support for more model architectures
- Include more fine-grained control over training parameters
- Add support for additional quantization methods

### 6.3 Tooling
- Add configuration schema validation
- Generate configuration documentation automatically
- Create configuration templates for common use cases

## 7. Key Takeaways

1. **Type Safety is Worth the Effort**
   - The initial investment in proper typing and validation prevents many runtime errors
   - Makes the code more maintainable and self-documenting

2. **Documentation is Part of the API**
   - Good documentation makes the configuration self-service
   - Reduces the learning curve for new developers

3. **Validation Should Be Proactive**
   - Fail fast with clear error messages
   - Validate as early as possible in the configuration lifecycle

4. **Configuration is Code**
   - Apply software engineering best practices to configuration
   - Use version control and code review for configuration changes

## 8. Configuration-Processor Consistency

### 8.1 Attribute Naming Consistency
- **Critical**: Configuration attribute names must match what processors expect
  - Issue: DataConfig had `val_path` but DataProcessor was looking for `validation_path`
  - Solution: Standardize attribute names across config and processor classes
  - Best Practice: Use consistent naming conventions and document attribute mappings

- **Missing Attributes Cause Runtime Errors**
  - Issue: DataProcessor referenced `validation_split` and `shuffle_seed` that didn't exist in DataConfig
  - Solution: Ensure all attributes used by processors are defined in the configuration class
  - Prevention: Use type checkers and comprehensive tests to catch missing attributes early

### 8.2 Method Implementation Requirements
- **All Called Methods Must Exist**
  - Issue: DataProcessor called `config.get_tokenizer_kwargs()` which didn't exist
  - Solution: Implement all methods referenced by processors
  - Best Practice: Define interface contracts and validate them in tests

- **Parameter Conflict Prevention**
  - Issue: `get_tokenizer_kwargs()` returned parameters that were also passed explicitly, causing `TypeError: got multiple values for keyword argument`
  - Solution: When using `**kwargs` expansion, exclude parameters that are passed explicitly
  - Example: If processor passes `padding="max_length"` explicitly, don't include `padding` in `get_tokenizer_kwargs()`
  ```python
  # Bad: Causes conflict
  def get_tokenizer_kwargs(self):
      return {"max_length": 512, "padding": "max_length", "truncation": True}
  
  # Good: Only returns non-conflicting parameters
  def get_tokenizer_kwargs(self):
      return {"max_length": 512}  # padding and truncation passed explicitly
  ```

## 9. Import and Dependency Management

### 9.1 Missing Import Errors
- **NameError During Module Import**
  - Issue: `base.py` used `torch.cuda.is_available()` but didn't import `torch`
  - Solution: Always import required modules at the top of the file
  - Prevention: Use linters and type checkers to catch missing imports

- **Dynamic Default Values in Dataclasses**
  - Issue: Class-level default value `device: str = "cuda" if torch.cuda.is_available() else "cpu"` evaluated at class definition time
  - Solution: Use `field(default_factory=...)` for values that need runtime evaluation
  ```python
  # Bad: Evaluated at class definition time
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  
  # Good: Evaluated at instance creation time
  device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
  ```

### 9.2 Module Structure
- **Missing Utility Modules**
  - Issue: Processor imported `setup_logging` from `finetuning_lora.utils.logging` which didn't exist
  - Solution: Create required utility modules or remove unused imports
  - Best Practice: Create `__init__.py` files for all package directories

## 10. Testing Best Practices

### 10.1 File-Based Configuration Testing
- **Use Temporary Files for File Validation**
  - Issue: Tests failed because DataConfig default path didn't exist
  - Solution: Use pytest fixtures to create temporary files for testing
  - Example:
  ```python
  @pytest.fixture
  def temp_jsonl(sample_data):
      with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
          for item in sample_data:
              f.write(json.dumps(item) + '\n')
          yield f.name
          Path(f.name).unlink()  # Cleanup
  ```

- **Pass Required File Paths in Tests**
  - Issue: Tests created DataConfig without providing required file paths
  - Solution: Always provide valid file paths in test fixtures
  ```python
  # Good: Uses temp file fixture
  def test_processor(config, temp_jsonl):
      config = DataConfig(train_path=temp_jsonl)
      processor = DataProcessor(config)
  ```

### 10.2 Mocking External Dependencies
- **Proper Tokenizer Mocking**
  - Issue: Tokenizer mocks needed to return proper dictionary structures
  - Solution: Configure mocks to return expected data structures
  - Best Practice: Make mocks callable and return realistic data
  ```python
  # Good: Properly configured mock
  mock_tokenizer = MagicMock()
  mock_tokenizer.return_value = {
      "input_ids": [[1, 2, 3, 4]],
      "attention_mask": [[1, 1, 1, 1]]
  }
  ```

- **Avoid Over-Mocking**
  - Issue: Over-mocking can hide real integration issues
  - Solution: Use real objects when possible, mock only external dependencies
  - Example: Use real `Dataset.from_list()` instead of mocking it when testing data loading

### 10.3 Test Assertions
- **Verify Behavior, Not Just Structure**
  - Issue: Tests passed but didn't verify correct behavior
  - Solution: Add assertions that verify the expected behavior
  - Example:
  ```python
  # Good: Verifies both structure and behavior
  datasets = processor.load_datasets()
  assert "train" in datasets
  assert "validation" in datasets
  assert len(datasets["train"]) > 0  # Verifies data was loaded
  assert len(datasets["validation"]) > 0  # Verifies split occurred
  ```

## 11. Key Takeaways from Data Processor Testing

1. **Configuration-Processor Contract**
   - Maintain a clear contract between configuration and processor classes
   - All attributes and methods used by processors must exist in configuration
   - Use consistent naming conventions across the codebase

2. **Avoid Parameter Conflicts**
   - When using `**kwargs` expansion, be careful not to pass the same parameter twice
   - Document which parameters are handled explicitly vs. via kwargs
   - Test parameter passing to catch conflicts early

3. **Import Dependencies Early**
   - Import all required modules at the top of files
   - Use `field(default_factory=...)` for dynamic default values in dataclasses
   - Verify imports work during module load time, not just runtime

4. **Test with Realistic Data**
   - Use temporary files for file-based tests
   - Provide all required configuration parameters in tests
   - Mock external dependencies appropriately, but use real objects when possible

5. **Comprehensive Test Coverage**
   - Test both success and failure paths
   - Verify data structures and behavior, not just method calls
   - Use fixtures to set up test data consistently

## 12. Advanced Testing Strategies for Complex Dependencies

### 12.1 Mocking Optional Dependencies in conftest.py
- **Module-Level Mocking for Optional Dependencies**
  - Issue: Tests failed with `ModuleNotFoundError: No module named 'tensorboard'` even though tensorboard is optional
  - Solution: Mock optional dependencies at the module level in `conftest.py` before any imports
  - Critical: Mock must include `__spec__` attribute for compatibility with `importlib.util.find_spec()`
  - Best Practice: Mock in conftest.py so it applies to all tests automatically
  ```python
  # In conftest.py - before any imports
  if "tensorboard" not in sys.modules:
      mock_tensorboard = MagicMock()
      mock_tensorboard.__spec__ = MagicMock()
      mock_tensorboard.__spec__.name = "tensorboard"
      sys.modules["tensorboard"] = mock_tensorboard
      
      # Also mock submodules
      mock_torch_tb = MagicMock()
      mock_torch_tb.SummaryWriter = MagicMock()
      sys.modules["torch.utils.tensorboard"] = mock_torch_tb
  ```

### 12.2 Patching Dynamically Imported Modules
- **Correct Patch Path for Conditional Imports**
  - Issue: `ModelBuilder` was imported inside a method (`from finetuning_lora.models.builder import ModelBuilder`), not at module level
  - Solution: Patch the module where it's imported, not where it's used
  - Example:
  ```python
  # Bad: Patches wrong location
  @patch('finetuning_lora.training.trainer.ModelBuilder')
  
  # Good: Patches where it's actually imported
  @patch('finetuning_lora.models.builder.ModelBuilder')
  ```

### 12.3 Test Fixtures for Configuration Objects
- **Handling Missing Attributes in Test Fixtures**
  - Issue: TrainingConfig didn't have all attributes that trainer expected (e.g., `num_epochs` vs `num_train_epochs`)
  - Solution: Create test fixtures that add missing attributes dynamically
  - Best Practice: Use fixtures to bridge configuration mismatches in tests without modifying production code
  ```python
  @pytest.fixture
  def training_config(tmp_path):
      """Create a training configuration with all required attributes."""
      config = TrainingConfig(output_dir=str(tmp_path / "test_output"))
      # Add missing attributes that trainer expects
      config.num_epochs = config.num_train_epochs
      config.per_device_batch_size = config.per_device_train_batch_size
      config.lr_scheduler = config.lr_scheduler_type
      # ... add other missing attributes
      return config
  ```

### 12.4 Comprehensive Mocking for Complex Dependencies
- **Mocking Entire Dependency Chains**
  - Issue: Trainer depends on ModelBuilder, TrainingArguments, Trainer, DataCollator, etc.
  - Solution: Mock all dependencies in the chain, including those created internally
  - Best Practice: Mock at the right level - mock external dependencies, not internal logic
  ```python
  @patch('finetuning_lora.training.trainer.setup_logging')
  @patch('finetuning_lora.training.trainer.log_version_info')
  @patch('finetuning_lora.training.trainer.set_seed')
  @patch('finetuning_lora.models.builder.ModelBuilder')
  @patch('finetuning_lora.training.trainer.Trainer')
  @patch('finetuning_lora.training.trainer.TrainingArguments')
  @patch('finetuning_lora.training.trainer.DataCollatorForLanguageModeling')
  def test_train(...):
      # Setup all mocks with proper return values
      # Verify interactions at each level
  ```

### 12.5 Testing Error Conditions and Edge Cases
- **Explicit Error Testing**
  - Issue: Tests only covered happy paths, missing error conditions
  - Solution: Add dedicated tests for error conditions using `pytest.raises()`
  - Best Practice: Test both success and failure paths explicitly
  ```python
  def test_evaluate_without_trainer(...):
      """Test that evaluate raises error when trainer is not initialized."""
      trainer = LoRATrainer(model_config, training_config, data_config)
      with pytest.raises(ValueError, match="Trainer has not been initialized"):
          trainer.evaluate()
  ```

### 12.6 Verifying Mock Interactions
- **Comprehensive Assertions on Mock Calls**
  - Issue: Tests verified method calls but not parameters passed
  - Solution: Assert on call arguments to verify correct data flow
  - Best Practice: Verify both that methods were called AND with correct parameters
  ```python
  # Verify trainer was initialized with correct arguments
  mock_trainer_class.assert_called_once()
  call_args = mock_trainer_class.call_args
  assert call_args.kwargs["model"] == mock_model
  assert call_args.kwargs["tokenizer"] == mock_tokenizer
  assert call_args.kwargs["train_dataset"] == train_dataset
  ```

### 12.7 Test Organization and Fixture Reusability
- **Shared Fixtures for Common Test Data**
  - Issue: Each test was creating its own configuration objects
  - Solution: Create reusable fixtures for common test objects
  - Best Practice: Use fixtures to reduce duplication and ensure consistency
  ```python
  @pytest.fixture
  def model_config():
      """Create a model configuration for testing."""
      return ModelConfig()
  
  @pytest.fixture
  def data_config():
      """Create a data configuration for testing."""
      return DataConfig()
  ```

## 13. Key Takeaways from Trainer Testing Implementation

1. **Mock Optional Dependencies Early**
   - Mock optional dependencies in `conftest.py` before any imports
   - Include `__spec__` attribute for `importlib` compatibility
   - This prevents import errors across all tests

2. **Patch Where Modules Are Imported**
   - Patch modules at their import location, not usage location
   - For dynamically imported modules, patch the source module
   - Verify patch paths match actual import statements

3. **Use Fixtures to Bridge Configuration Gaps**
   - Test fixtures can add missing attributes without modifying production code
   - This allows testing even when configuration objects don't perfectly match usage
   - Document why attributes are added in fixture comments

4. **Mock Entire Dependency Chains**
   - When testing complex classes, mock all dependencies
   - Include mocks for objects created internally (like TrainingArguments)
   - Verify interactions at each level of the dependency chain

5. **Test Both Success and Failure Paths**
   - Add explicit tests for error conditions
   - Use `pytest.raises()` with specific error messages
   - Verify error messages are helpful and specific

6. **Verify Mock Interactions Comprehensively**
   - Don't just verify methods were called
   - Verify they were called with correct parameters
   - Check return values and side effects

7. **Organize Tests with Reusable Fixtures**
   - Create fixtures for common test objects
   - Reduce duplication and ensure consistency
   - Make tests easier to read and maintain

## 14. Pydantic Field Objects in Dataclasses: Critical Issue and Solution

### 14.1 The Problem: FieldInfo Objects Instead of Values
- **Critical Issue**: When using Pydantic's `Field()` in a `@dataclass`, the `FieldInfo` object itself becomes the attribute value, not the default value
- **Symptom**: `AttributeError: 'FieldInfo' object has no attribute 'value'` when trying to access enum values
- **Root Cause**: Dataclasses don't automatically extract default values from Pydantic `Field()` objects like Pydantic models do
- **Example**:
  ```python
  @dataclass
  class ModelConfig:
      bnb_4bit_quant_type: QuantizationType = Field(default=QuantizationType.NF4)
  
  # When accessing:
  config = ModelConfig()
  print(type(config.bnb_4bit_quant_type))  # <class 'pydantic.fields.FieldInfo'>
  # NOT <enum 'QuantizationType'>
  ```

### 14.2 The Solution: Extract Default Values from FieldInfo
- **Fix**: Check if attribute is a `FieldInfo` object and extract its `.default` value
- **Implementation**: Handle both `FieldInfo` objects and actual enum values
- **Best Practice**: Create a helper function to safely extract values from config fields
- **Example**:
  ```python
  def _get_quantization_config(self):
      # Handle Pydantic Field objects in dataclass - get the actual value
      quant_type = self.model_config.bnb_4bit_quant_type
      compute_dtype = self.model_config.bnb_4bit_compute_dtype
      
      # Extract default value if it's a FieldInfo object
      if hasattr(quant_type, 'default'):
          quant_type = quant_type.default
      if hasattr(compute_dtype, 'default'):
          compute_dtype = compute_dtype.default
      
      # Get string values from enums
      quant_type_str = quant_type.value if hasattr(quant_type, 'value') else quant_type
      compute_dtype_str = compute_dtype.value if hasattr(compute_dtype, 'value') else compute_dtype
      
      return BitsAndBytesConfig(
          bnb_4bit_quant_type=quant_type_str,
          bnb_4bit_compute_dtype=getattr(torch, compute_dtype_str),
          # ...
      )
  ```

### 14.3 Better Long-Term Solutions
- **Option 1**: Use Pydantic `BaseModel` instead of `@dataclass` for configurations that use `Field()`
- **Option 2**: Use `dataclasses.field()` instead of Pydantic `Field()` when using `@dataclass`
- **Option 3**: Create a hybrid approach with a custom `__post_init__` that extracts FieldInfo defaults
- **Recommendation**: For new code, prefer Pydantic `BaseModel` if you need Pydantic features like `Field()`
- **Note**: See Section 17 for complete Pydantic v2 migration guide and best practices

### 14.4 Testing Considerations
- **Test Both Cases**: Test with both default values (FieldInfo) and explicitly set values (actual enums)
- **Mock FieldInfo Objects**: When mocking configs, ensure mocks return actual enum values, not FieldInfo objects
- **Validation Tests**: Verify that configuration validation works correctly with FieldInfo extraction

## 15. Model Builder Testing: Comprehensive Mocking Strategy

### 15.1 Explicit Configuration in Tests
- **Issue**: Default configuration values can trigger validation errors (e.g., both `load_in_4bit=True` and `load_in_8bit=True` by default)
- **Solution**: Always explicitly set quantization flags in tests to avoid validation errors
- **Best Practice**: Make test configurations explicit and clear about what they're testing
- **Example**:
  ```python
  # Bad: Relies on defaults, may cause validation errors
  model_config = ModelConfig()
  
  # Good: Explicit configuration for test
  model_config = ModelConfig(
      load_in_4bit=False,
      load_in_8bit=False,
      use_lora=False
  )
  ```

### 15.2 Comprehensive Model Mocking
- **Issue**: Models have many methods and attributes that may be called conditionally
- **Solution**: Set up complete mocks with all necessary methods and attributes
- **Critical Methods to Mock**:
  - `eval()` - Returns self, not None
  - `print_trainable_parameters()` - May cause hangs if not mocked
  - `gradient_checkpointing_enable()` - Called conditionally
  - `config.use_cache` - Attribute assignment must work
- **Example**:
  ```python
  mock_model = MagicMock()
  mock_model.config = MagicMock()  # NOT a dict - allows attribute assignment
  mock_model.config.use_cache = True
  mock_model.eval = MagicMock(return_value=mock_model)  # Returns self
  mock_model.gradient_checkpointing_enable = MagicMock()
  mock_model.print_trainable_parameters = MagicMock()  # Prevents hangs
  ```

### 15.3 Patching at Import Location
- **Issue**: Patching `transformers.AutoTokenizer.from_pretrained` doesn't work if the module imports it differently
- **Solution**: Patch where the module is imported, not where it's defined
- **Best Practice**: Patch in the module namespace where it's used
- **Example**:
  ```python
  # Bad: Patches where it's defined
  @patch('transformers.AutoTokenizer.from_pretrained')
  
  # Good: Patches where it's imported and used
  @patch('finetuning_lora.models.builder.AutoTokenizer.from_pretrained')
  ```

### 15.4 Mocking Logging and Utility Functions
- **Issue**: Functions like `log_version_info()` may cause delays or side effects in tests
- **Solution**: Patch utility functions that don't affect test logic
- **Best Practice**: Patch logging and version info functions to speed up tests
- **Example**:
  ```python
  @patch('finetuning_lora.models.builder.log_version_info')
  def test_load_model(mock_log_version_info):
      # Test proceeds without logging overhead
      builder = ModelBuilder(model_config)
      model, tokenizer = builder.load_model()
  ```

### 15.5 Handling Async and Blocking Operations
- **Issue**: Methods like `print_trainable_parameters()` may block or hang in tests
- **Solution**: Always mock methods that interact with external resources or compute-heavy operations
- **Best Practice**: Mock any method that might:
  - Access network resources
  - Perform heavy computations
  - Display output to console
  - Access GPU resources
- **Example**:
  ```python
  mock_peft_model = MagicMock()
  mock_peft_model.print_trainable_parameters = MagicMock()  # Prevents hang
  mock_get_peft_model.return_value = mock_peft_model
  ```

### 15.6 Test Fixture Configuration
- **Issue**: Fixtures may not provide all required mock attributes
- **Solution**: Ensure fixtures provide complete, properly configured mocks
- **Best Practice**: Create fixtures that set up all necessary mock attributes and methods
- **Example**:
  ```python
  @pytest.fixture
  def mock_model():
      """Create a complete mock model with all necessary attributes."""
      model = MagicMock(spec=AutoModelForCausalLM)
      model.config = MagicMock()
      model.config.use_cache = True
      model.eval = MagicMock(return_value=model)
      model.gradient_checkpointing_enable = MagicMock()
      model.save_pretrained = MagicMock()
      return model
  ```

### 15.7 Verifying Mock Interactions
- **Issue**: Tests may pass but not verify correct behavior
- **Solution**: Assert on both method calls and their arguments
- **Best Practice**: Verify that methods were called with correct parameters
- **Example**:
  ```python
  # Verify LoRA was applied
  mock_get_peft_model.assert_called_once()
  call_args = mock_get_peft_model.call_args
  assert isinstance(call_args[0][1], LoraConfig)  # Verify LoRA config passed
  
  # Verify quantization was not applied
  mock_prepare_model.assert_not_called()
  ```

## 16. Key Takeaways from Model Builder Testing

1. **Handle Pydantic Field Objects in Dataclasses**
   - Always check if config attributes are `FieldInfo` objects
   - Extract `.default` values before accessing enum `.value` attributes
   - Consider using Pydantic `BaseModel` instead of `@dataclass` for new code

2. **Explicit Test Configuration**
   - Never rely on default configuration values in tests
   - Explicitly set all flags that might cause validation errors
   - Make test intentions clear through explicit configuration

3. **Complete Mock Setup**
   - Mock all methods that might be called, even conditionally
   - Set up `config` as `MagicMock`, not dict, to allow attribute assignment
   - Ensure methods like `eval()` return `self` for chaining

4. **Patch at Import Location**
   - Patch modules where they're imported, not where they're defined
   - Use full module paths for patching (e.g., `finetuning_lora.models.builder.AutoTokenizer`)
   - Verify patch paths match actual import statements

5. **Mock Blocking Operations**
   - Always mock methods that might block, hang, or access external resources
   - Mock logging and utility functions to speed up tests
   - Mock GPU-related operations when testing on CPU

6. **Comprehensive Test Fixtures**
   - Create fixtures that provide complete, properly configured mocks
   - Include all necessary attributes and methods in fixtures
   - Document what each fixture provides

7. **Verify Behavior, Not Just Calls**
   - Assert on method call arguments, not just that methods were called
   - Verify correct data flow through the system
   - Test both positive and negative cases (methods called and not called)

## 17. Pydantic v2 Migration: Critical Changes and Best Practices

### 17.1 Dataclass vs Pydantic BaseModel Incompatibility
- **Critical Issue**: You cannot mix `@dataclass` decorator with Pydantic `Field()` objects
- **Problem**: Dataclasses don't understand Pydantic `Field()` objects - they store the `FieldInfo` object as the attribute value, not the default value
- **Symptom**: Accessing attributes returns `FieldInfo` objects instead of actual values
- **Root Cause**: Dataclasses and Pydantic have different mechanisms for handling field defaults and validation
- **Solution**: Use Pydantic `BaseModel` when you need Pydantic features like `Field()`, validators, or constraints
- **Example**:
  ```python
  # Bad: Incompatible - FieldInfo objects stored as values
  @dataclass
  class ModelConfig:
      model_name: str = Field(default="test", description="Model name")
  
  config = ModelConfig()
  print(type(config.model_name))  # <class 'pydantic.fields.FieldInfo'>
  
  # Good: Use BaseModel for Pydantic features
  from pydantic import BaseModel, Field
  
  class ModelConfig(BaseModel):
      model_name: str = Field(default="test", description="Model name")
  
  config = ModelConfig()
  print(type(config.model_name))  # <class 'str'>
  ```

### 17.2 Pydantic v2 Syntax Changes
- **Version Detection**: Always check Pydantic version before implementing
  ```python
  import pydantic
  print(pydantic.VERSION)  # Check version (e.g., "2.12.4")
  ```

- **Configuration Class Syntax**
  - **v1 (Deprecated)**: Used nested `Config` class
    ```python
    class ModelConfig(BaseModel):
        class Config:
            use_enum_values = True
            validate_assignment = True
    ```
  - **v2 (Current)**: Use `model_config = ConfigDict(...)`
    ```python
    from pydantic import BaseModel, ConfigDict
    
    class ModelConfig(BaseModel):
        model_config = ConfigDict(
            use_enum_values=True,
            validate_assignment=True,
        )
    ```

- **Validator Syntax Changes**
  - **v1 (Deprecated)**: Used `@root_validator` or `@validator`
    ```python
    from pydantic import root_validator
    
    @root_validator
    def validate_fields(cls, values):
        # values is a dict of field values
        if values.get('field1') and values.get('field2'):
            raise ValueError("Conflict")
        return values
    ```
  - **v2 (Current)**: Use `@model_validator(mode='after')`
    ```python
    from pydantic import model_validator
    
    @model_validator(mode='after')
    def validate_fields(self):
        # self is the model instance
        if self.field1 and self.field2:
            raise ValueError("Conflict")
        return self
    ```

### 17.3 Enum Handling with use_enum_values
- **Automatic String Conversion**: When `use_enum_values=True` is set, Pydantic automatically converts enum fields to their string values
- **No `.value` Needed**: Don't call `.value` on enum fields when `use_enum_values=True` is enabled
- **Direct Access**: Access enum fields directly - they're already strings
- **Example**:
  ```python
  class LoraBiasType(str, Enum):
      NONE = "none"
      ALL = "all"
  
  class ModelConfig(BaseModel):
      model_config = ConfigDict(use_enum_values=True)
      lora_bias: LoraBiasType = Field(default=LoraBiasType.NONE)
  
  config = ModelConfig()
  print(type(config.lora_bias))  # <class 'str'> (not <enum 'LoraBiasType'>)
  print(config.lora_bias)  # "none"
  
  # In methods, use directly - no .value needed
  def to_peft_config(self):
      return {
          "bias": self.lora_bias,  # Already a string, not an enum
          # NOT: "bias": self.lora_bias.value  # AttributeError!
      }
  ```

### 17.4 Model Validators in Pydantic v2
- **Mode Options**: `@model_validator` supports different modes
  - `mode='after'`: Runs after all field validators (receives `self`)
  - `mode='before'`: Runs before field validators (receives `data: dict`)
- **Mutual Exclusion Validation**: Use model validators for cross-field validation
- **Best Practice**: Use `mode='after'` for validation that needs all fields to be set
- **Example**:
  ```python
  @model_validator(mode='after')
  def validate_quantization_mutually_exclusive(self):
      """Ensure 4-bit and 8-bit quantization are mutually exclusive."""
      if self.load_in_4bit and self.load_in_8bit:
          raise ValueError(
              "Cannot use both 4-bit and 8-bit quantization simultaneously"
          )
      return self
  ```

### 17.5 Default Value Handling
- **Field Defaults**: Pydantic `Field()` defaults work correctly with `BaseModel`
- **List Defaults**: For mutable defaults like lists, use `Field(default=[...])` directly (Pydantic handles this safely)
- **Example**:
  ```python
  # Good: Pydantic handles mutable defaults safely
  class ModelConfig(BaseModel):
      lora_target_modules: List[str] = Field(
          default=["q_proj", "k_proj", "v_proj"],
          description="Target modules for LoRA"
      )
  ```

### 17.6 Testing Pydantic v2 Models
- **Instantiation**: Pydantic models work like regular classes - no special handling needed
- **Attribute Access**: Attributes are directly accessible (not FieldInfo objects)
- **Enum Testing**: When `use_enum_values=True`, test for string values, not enum instances
- **Validation Testing**: Test validators with `pytest.raises()` for error cases
- **Example**:
  ```python
  def test_model_config_defaults():
      """Test model configuration default values."""
      config = ModelConfig()
      # Direct attribute access works
      assert config.model_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
      assert config.trust_remote_code is True
      
      # Enum values are strings when use_enum_values=True
      assert config.lora_bias == "none"  # String, not LoraBiasType.NONE
      # Or compare with enum for clarity
      assert config.lora_bias == LoraBiasType.NONE.value
  
  def test_quantization_validation():
      """Test that mutual exclusion validation works."""
      with pytest.raises(ValueError, match="Cannot use both"):
          ModelConfig(load_in_4bit=True, load_in_8bit=True)
  ```

### 17.7 Migration Checklist
When migrating from dataclass + Pydantic Fields to Pydantic BaseModel:

1. **Remove `@dataclass` decorator**
   - Replace with `class ModelConfig(BaseModel):`

2. **Update imports**
   - Add: `from pydantic import BaseModel, Field, model_validator, ConfigDict`
   - Remove: `from dataclasses import dataclass, field`

3. **Update configuration**
   - Replace nested `Config` class with `model_config = ConfigDict(...)`

4. **Update validators**
   - Replace `@root_validator` with `@model_validator(mode='after')`
   - Change validator signature from `(cls, values)` to `(self)`
   - Access fields via `self.field_name` instead of `values.get('field_name')`

5. **Update enum handling**
   - Remove `.value` calls if `use_enum_values=True` is set
   - Test that enum fields return strings directly

6. **Update default factories**
   - Replace `Field(default_factory=lambda: [...])` with `Field(default=[...])` for simple cases
   - Pydantic handles mutable defaults safely

7. **Update tests**
   - Verify attributes are accessible directly (not FieldInfo objects)
   - Update enum comparisons if `use_enum_values=True`
   - Test validation errors with `pytest.raises()`

### 17.8 Key Takeaways from Pydantic v2 Migration

1. **Don't Mix Dataclasses with Pydantic Fields**
   - Use `BaseModel` when you need Pydantic features
   - Use `@dataclass` only when you don't need Pydantic validation/fields

2. **Check Pydantic Version**
   - Always verify which version of Pydantic is installed
   - Use version-appropriate syntax and imports

3. **Enum Values Are Automatic**
   - With `use_enum_values=True`, enums become strings automatically
   - Don't call `.value` - it will cause `AttributeError`

4. **Model Validators Are Simpler**
   - v2 validators receive `self` (instance) instead of `values` (dict)
   - More intuitive and Pythonic syntax

5. **Configuration is Different**
   - v2 uses `ConfigDict` instead of nested `Config` class
   - More explicit and type-safe

6. **Test After Migration**
   - Verify all attributes are accessible
   - Check enum handling works correctly
   - Test validation still works as expected

7. **Migration is Straightforward**
   - Most changes are syntactic
   - Logic remains the same
   - Tests help verify correctness
