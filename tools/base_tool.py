"""
DataNeuron Extensible Tool Architecture Foundation
==================================================

This module provides the base architecture for all tools in the DataNeuron project.
It defines a standardized interface that ensures all tools can be managed consistently
by the ToolManager while maintaining type safety and validation.

Features:
- Abstract base class for tool standardization
- Pydantic schema validation for inputs and outputs
- Consistent error handling and reporting
- Plug-and-play architecture for easy tool integration
- Runtime type checking and validation

Usage:
    from tools.base_tool import BaseTool
    from pydantic import BaseModel
    
    class MyToolArgs(BaseModel):
        text: str
        max_length: int = 100
    
    class MyToolResult(BaseModel):
        result: str
        metadata: dict
    
    class MyTool(BaseTool):
        name = "my_tool"
        description = "This tool processes text and returns results"
        args_schema = MyToolArgs
        return_schema = MyToolResult
        
        def _execute(self, **kwargs) -> dict:
            # Tool-specific logic here
            return {"result": "processed", "metadata": {}}

Architecture:
- All tools inherit from BaseTool
- Input validation via Pydantic args_schema
- Output validation via Pydantic return_schema
- Standardized error handling and reporting
- Thread-safe execution wrapper
"""

from abc import ABC, abstractmethod
from typing import Any, Type, Dict, Optional
from pydantic import BaseModel, ValidationError
import traceback


# ============================================================================
# BASE PYDANTIC MODELS FOR TOOL SCHEMAS
# ============================================================================

class BaseToolArgs(BaseModel):
    """
    Base class for tool argument schemas.
    
    All tool argument schemas should inherit from this to ensure consistency.
    """
    class Config:
        # Allow extra fields for flexibility
        extra = "forbid"  # Strict validation - no extra fields allowed
        # Validate field types strictly
        validate_assignment = True


class BaseToolResult(BaseModel):
    """
    Base class for tool result schemas.
    
    All tool result schemas should inherit from this to ensure consistency.
    """
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        # Allow extra fields for tool-specific results
        extra = "allow"
        validate_assignment = True


class ToolError(BaseToolResult):
    """
    Standard error result format for tool execution failures.
    """
    success: bool = False
    error_type: str
    error_details: Optional[str] = None
    
    class Config:
        extra = "allow"


# ============================================================================
# ABSTRACT BASE TOOL CLASS
# ============================================================================

class BaseTool(ABC):
    """
    Abstract base class for all tools in the DataNeuron system.
    
    This class provides a standardized interface for tool execution with:
    - Input/output validation using Pydantic schemas
    - Consistent error handling and reporting
    - Metadata tracking and logging capabilities
    - Thread-safe execution wrapper
    
    Attributes:
        name (str): Unique identifier for the tool
        description (str): Detailed description of tool functionality and usage
        args_schema (Type[BaseModel]): Pydantic model for input validation
        return_schema (Type[BaseModel]): Pydantic model for output validation
        version (str): Tool version for compatibility tracking
        category (str): Tool category for organization
    """
    
    # Required attributes that must be defined by concrete implementations
    name: str = None
    description: str = None
    args_schema: Type[BaseModel] = None
    return_schema: Type[BaseModel] = None
    
    # Optional attributes with defaults
    version: str = "1.0.0"
    category: str = "general"
    requires_session: bool = False  # Whether tool needs session context
    is_async: bool = False  # Whether tool supports async execution
    
    def __init__(self):
        """
        Initialize the tool and validate required attributes.
        
        Raises:
            ValueError: If required attributes are not properly defined
        """
        self._validate_tool_definition()
    
    def _validate_tool_definition(self) -> None:
        """
        Validate that all required attributes are properly defined.
        
        Raises:
            ValueError: If any required attribute is missing or invalid
        """
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} must define a 'name' attribute")
        
        if not self.description:
            raise ValueError(f"{self.__class__.__name__} must define a 'description' attribute")
        
        if not self.args_schema or not issubclass(self.args_schema, BaseModel):
            raise ValueError(f"{self.__class__.__name__} must define a valid 'args_schema' (Pydantic BaseModel)")
        
        if not self.return_schema or not issubclass(self.return_schema, BaseModel):
            raise ValueError(f"{self.__class__.__name__} must define a valid 'return_schema' (Pydantic BaseModel)")
    
    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        """
        Execute the tool's core functionality.
        
        This method must be implemented by all concrete tool classes.
        It contains the actual business logic of the tool.
        
        Args:
            **kwargs: Validated arguments as defined by args_schema
            
        Returns:
            Any: Tool-specific result that matches return_schema
            
        Raises:
            Any tool-specific exceptions
        """
        pass
    
    def execute(self, **kwargs) -> BaseModel:
        """
        Public execution wrapper with validation and error handling.
        
        This method provides a standardized execution interface that:
        1. Validates input arguments against args_schema
        2. Calls the tool's _execute method
        3. Validates output against return_schema
        4. Handles errors consistently
        
        Args:
            **kwargs: Raw input arguments to be validated
            
        Returns:
            BaseModel: Validated result matching return_schema or ToolError
        """
        try:
            # Step 1: Validate input arguments
            try:
                validated_args = self.args_schema(**kwargs)
                # Convert back to dict for _execute method
                execution_kwargs = validated_args.model_dump()
                
            except ValidationError as e:
                return ToolError(
                    error_type="ValidationError",
                    error_message=f"Invalid input arguments for tool '{self.name}'",
                    error_details=str(e),
                    metadata={
                        "tool_name": self.name,
                        "validation_errors": e.errors(),
                        "input_data": kwargs
                    }
                )
            
            # Step 2: Execute the tool's core logic
            try:
                raw_result = self._execute(**execution_kwargs)
                
            except Exception as e:
                return ToolError(
                    error_type=type(e).__name__,
                    error_message=f"Tool execution failed for '{self.name}': {str(e)}",
                    error_details=traceback.format_exc(),
                    metadata={
                        "tool_name": self.name,
                        "input_args": execution_kwargs,
                        "exception_type": type(e).__name__
                    }
                )
            
            # Step 3: Validate and format output
            try:
                # If raw_result is already a dict, use it directly
                if isinstance(raw_result, dict):
                    result_data = raw_result
                else:
                    # If it's a simple value, wrap it in a standard format
                    result_data = {"result": raw_result}
                
                # Ensure success flag is set
                if "success" not in result_data:
                    result_data["success"] = True
                
                # Validate against return schema
                validated_result = self.return_schema(**result_data)
                return validated_result
                
            except ValidationError as e:
                return ToolError(
                    error_type="OutputValidationError",
                    error_message=f"Tool '{self.name}' produced invalid output format",
                    error_details=str(e),
                    metadata={
                        "tool_name": self.name,
                        "raw_result": str(raw_result)[:500],  # Limit size
                        "validation_errors": e.errors()
                    }
                )
                
        except Exception as e:
            # Catch-all for unexpected errors
            return ToolError(
                error_type="UnexpectedError",
                error_message=f"Unexpected error in tool '{self.name}': {str(e)}",
                error_details=traceback.format_exc(),
                metadata={
                    "tool_name": self.name,
                    "input_kwargs": str(kwargs)[:200]  # Limit size
                }
            )
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the tool's schemas and capabilities.
        
        Returns:
            Dict containing tool metadata, input/output schemas, and examples
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "requires_session": self.requires_session,
            "is_async": self.is_async,
            "input_schema": self.args_schema.model_json_schema() if self.args_schema else None,
            "output_schema": self.return_schema.model_json_schema() if self.return_schema else None,
        }
    
    def validate_args(self, **kwargs) -> BaseModel:
        """
        Validate arguments without executing the tool.
        
        Useful for testing and validation purposes.
        
        Args:
            **kwargs: Arguments to validate
            
        Returns:
            Validated arguments as Pydantic model
            
        Raises:
            ValidationError: If arguments are invalid
        """
        return self.args_schema(**kwargs)
    
    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
    
    def __repr__(self) -> str:
        """Developer representation of the tool."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"description='{self.description[:50]}...', version='{self.version}')")


# ============================================================================
# EXAMPLE TOOL IMPLEMENTATION FOR TESTING
# ============================================================================

class ExampleAdditionToolArgs(BaseToolArgs):
    """Arguments for the example addition tool."""
    a: float
    b: float
    precision: int = 2


class ExampleAdditionToolResult(BaseToolResult):
    """Result for the example addition tool."""
    result: float
    operation: str
    inputs: Dict[str, float]


class ExampleAdditionTool(BaseTool):
    """
    Example tool implementation for testing the BaseTool architecture.
    
    This tool adds two numbers together with configurable precision.
    """
    
    name = "add_numbers"
    description = (
        "Adds two numbers together with configurable decimal precision. "
        "This tool accepts two numeric inputs (a and b) and returns their sum "
        "rounded to the specified number of decimal places. "
        "Use this tool when you need to perform basic arithmetic addition "
        "with precise decimal control."
    )
    args_schema = ExampleAdditionToolArgs
    return_schema = ExampleAdditionToolResult
    version = "1.0.0"
    category = "math"
    
    def _execute(self, a: float, b: float, precision: int = 2) -> Dict[str, Any]:
        """
        Add two numbers with specified precision.
        
        Args:
            a: First number to add
            b: Second number to add
            precision: Number of decimal places for result
            
        Returns:
            Dictionary with addition result and metadata
        """
        result = round(a + b, precision)
        
        return {
            "result": result,
            "operation": f"{a} + {b}",
            "inputs": {"a": a, "b": b},
            "metadata": {
                "precision_used": precision,
                "tool_version": self.version
            }
        }


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the BaseTool architecture with various scenarios.
    This demonstrates proper usage and validates the error handling system.
    """
    
    print("=== DataNeuron BaseTool Architecture Test ===")
    
    # Test 1: Create a valid tool instance
    print("\nTest 1: Creating valid tool instance")
    try:
        addition_tool = ExampleAdditionTool()
        print(f" PASS - Tool created: {addition_tool}")
        print(f"   Schema info: {addition_tool.get_schema_info()['name']}")
    except Exception as e:
        print(f"L FAIL - Tool creation failed: {e}")
    
    # Test 2: Execute with valid arguments
    print("\nTest 2: Executing with valid arguments")
    try:
        result = addition_tool.execute(a=5.5, b=3.2, precision=1)
        print(f" PASS - Execution successful")
        print(f"   Result: {result.result}")
        print(f"   Success: {result.success}")
        print(f"   Metadata: {result.metadata}")
    except Exception as e:
        print(f"L FAIL - Valid execution failed: {e}")
    
    # Test 3: Execute with invalid arguments (missing required field)
    print("\nTest 3: Executing with invalid arguments (missing field)")
    try:
        result = addition_tool.execute(a=5.5)  # Missing 'b'
        print(f" PASS - Error handled gracefully")
        print(f"   Success: {result.success}")
        print(f"   Error: {result.error_message}")
        print(f"   Error Type: {result.error_type}")
    except Exception as e:
        print(f"L FAIL - Error handling failed: {e}")
    
    # Test 4: Execute with wrong argument types
    print("\nTest 4: Executing with wrong argument types")
    try:
        result = addition_tool.execute(a="not_a_number", b=3.2)
        print(f" PASS - Type error handled gracefully")
        print(f"   Success: {result.success}")
        print(f"   Error: {result.error_message}")
        print(f"   Validation errors: {len(result.metadata.get('validation_errors', []))}")
    except Exception as e:
        print(f"L FAIL - Type error handling failed: {e}")
    
    # Test 5: Argument validation without execution
    print("\nTest 5: Argument validation without execution")
    try:
        validated_args = addition_tool.validate_args(a=10.0, b=20.0, precision=3)
        print(f" PASS - Arguments validated successfully")
        print(f"   Validated args: a={validated_args.a}, b={validated_args.b}, precision={validated_args.precision}")
    except ValidationError as e:
        print(f"L FAIL - Argument validation failed: {e}")
    except Exception as e:
        print(f"L FAIL - Unexpected error: {e}")
    
    # Test 6: Invalid argument validation
    print("\nTest 6: Invalid argument validation")
    try:
        addition_tool.validate_args(a=10.0)  # Missing 'b'
        print(f"L FAIL - Should have raised ValidationError")
    except ValidationError as e:
        print(f" PASS - ValidationError correctly raised")
        print(f"   Error details: {len(e.errors())} validation errors")
    except Exception as e:
        print(f"L FAIL - Wrong exception type: {e}")
    
    # Test 7: Try to create tool with missing required attributes
    print("\nTest 7: Creating tool with missing attributes")
    try:
        class IncompleteTool(BaseTool):
            # Missing required attributes
            pass
        
        incomplete_tool = IncompleteTool()
        print(f"L FAIL - Should have raised ValueError")
    except ValueError as e:
        print(f" PASS - Tool validation correctly rejected incomplete tool")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"L FAIL - Wrong exception type: {e}")
    
    # Test 8: Schema information retrieval
    print("\nTest 8: Schema information retrieval")
    try:
        schema_info = addition_tool.get_schema_info()
        print(f" PASS - Schema info retrieved successfully")
        print(f"   Tool name: {schema_info['name']}")
        print(f"   Category: {schema_info['category']}")
        print(f"   Input schema keys: {list(schema_info['input_schema']['properties'].keys())}")
        print(f"   Output schema keys: {list(schema_info['output_schema']['properties'].keys())}")
    except Exception as e:
        print(f"L FAIL - Schema info retrieval failed: {e}")
    
    print("\n=== Test Complete ===")
    print("All BaseTool architecture tests completed.")
    print("The tool system is ready for concrete tool implementations.")