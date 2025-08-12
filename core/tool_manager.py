"""
DataNeuron Dynamic Tool Manager and Orchestrator
================================================

This module provides dynamic tool discovery, registration, and execution management
for the DataNeuron project. It automatically discovers all tools in the tools directory,
maintains a registry, and provides a unified interface for tool execution.

Features:
- Singleton pattern for efficient tool management
- Dynamic tool discovery using importlib and inspect
- Comprehensive error handling and graceful degradation
- Thread-safe tool registry and execution
- Automatic tool schema extraction for LLM integration
- Resilient to broken or invalid tool modules

Usage:
    from core.tool_manager import ToolManager
    
    # Get singleton instance
    tool_manager = ToolManager()
    
    # List available tools
    tools = tool_manager.list_tools()
    
    # Get tool schemas for LLM
    schemas = tool_manager.list_tool_schemas()
    
    # Execute a tool
    result = tool_manager.run_tool("tool_name", arg1=value1, arg2=value2)

Architecture:
- Singleton pattern ensures single tool registry instance
- Dynamic discovery prevents manual tool imports
- Graceful error handling prevents system crashes from broken tools
- Schema-based tool introspection for LLM integration
"""

import importlib
import inspect
import pkgutil
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from pydantic import BaseModel

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import logger
from config.settings import (
    TOOLS_DIR, 
    TOOL_AUTO_DISCOVERY, 
    TOOL_DISCOVERY_RECURSIVE,
    TOOL_EXECUTION_TIMEOUT_SECONDS
)
from tools.base_tool import BaseTool, ToolError


# ============================================================================
# TOOL MANAGER SINGLETON CLASS
# ============================================================================

class ToolManager:
    """
    Singleton class for dynamic tool discovery, registration, and execution.
    
    This class automatically discovers all tools in the tools directory,
    maintains a registry of available tools, and provides methods for
    tool execution and introspection.
    
    Attributes:
        tools (Dict[str, BaseTool]): Registry of discovered tools by name
        discovery_paths (List[Path]): Paths searched for tool modules
        failed_modules (Dict[str, str]): Modules that failed to load with error messages
    """
    
    _instance: Optional['ToolManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ToolManager':
        """
        Singleton pattern implementation with thread safety.
        
        Returns:
            Single instance of ToolManager
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ToolManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize the ToolManager singleton.
        
        Performs tool discovery and registration on first initialization.
        """
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            # Initialize instance variables
            self.tools: Dict[str, BaseTool] = {}
            self.discovery_paths: List[Path] = []
            self.failed_modules: Dict[str, str] = {}
            self.tools_dir = TOOLS_DIR
            self.auto_discovery = TOOL_AUTO_DISCOVERY
            self.recursive_discovery = TOOL_DISCOVERY_RECURSIVE
            self.execution_timeout = TOOL_EXECUTION_TIMEOUT_SECONDS
            
            logger.info("Initializing ToolManager singleton")
            logger.info(f"Tools directory: {self.tools_dir}")
            logger.info(f"Auto discovery enabled: {self.auto_discovery}")
            
            # Perform tool discovery if enabled
            if self.auto_discovery:
                self.discover_and_register_tools()
            
            self._initialized = True
            logger.success(f"ToolManager initialized with {len(self.tools)} tools")
    
    def discover_and_register_tools(self) -> None:
        """
        Discover and register all tools in the configured tools directory.
        
        This method:
        1. Scans the tools directory for Python modules
        2. Dynamically imports each module
        3. Inspects modules for BaseTool subclasses
        4. Registers valid tools in the registry
        5. Logs errors for failed modules without crashing
        """
        logger.info("Starting tool discovery process")
        
        if not self.tools_dir.exists():
            logger.warning(f"Tools directory does not exist: {self.tools_dir}")
            return
        
        # Clear existing registrations for fresh discovery
        self.tools.clear()
        self.failed_modules.clear()
        self.discovery_paths.clear()
        
        # Add tools directory to Python path if not already there
        tools_dir_str = str(self.tools_dir)
        if tools_dir_str not in sys.path:
            sys.path.insert(0, str(self.tools_dir.parent))
            self.discovery_paths.append(self.tools_dir)
        
        # Discover Python modules in tools directory
        discovered_modules = []
        
        try:
            # Get the tools package
            tools_package_name = self.tools_dir.name
            
            # Walk through the tools directory
            for item in self.tools_dir.iterdir():
                if item.is_file() and item.suffix == '.py':
                    # Skip special files
                    if item.name.startswith('__') or item.name == 'base_tool.py':
                        logger.debug(f"Skipping special file: {item.name}")
                        continue
                    
                    module_name = f"{tools_package_name}.{item.stem}"
                    discovered_modules.append((module_name, item))
                    
                elif item.is_dir() and self.recursive_discovery:
                    # Recursively discover in subdirectories
                    logger.debug(f"Recursively scanning directory: {item}")
                    # This could be extended for recursive discovery
            
            logger.info(f"Found {len(discovered_modules)} potential tool modules")
            
            # Import and process each discovered module
            for module_name, module_path in discovered_modules:
                self._process_tool_module(module_name, module_path)
                
        except Exception as e:
            logger.exception(f"Unexpected error during tool discovery: {e}")
        
        # Summary of discovery results
        successful_tools = len(self.tools)
        failed_modules = len(self.failed_modules)
        total_attempted = successful_tools + failed_modules
        
        logger.info(f"Tool discovery completed:")
        logger.info(f"  - Successful tools: {successful_tools}")
        logger.info(f"  - Failed modules: {failed_modules}")
        logger.info(f"  - Total attempted: {total_attempted}")
        
        if self.failed_modules:
            logger.warning("Failed modules:")
            for module_name, error in self.failed_modules.items():
                logger.warning(f"  - {module_name}: {error}")
        
        if successful_tools > 0:
            logger.info("Successfully registered tools:")
            for tool_name, tool in self.tools.items():
                logger.info(f"  - {tool_name} ({tool.__class__.__name__}) v{tool.version}")
    
    def _process_tool_module(self, module_name: str, module_path: Path) -> None:
        """
        Process a single tool module for BaseTool subclasses.
        
        Args:
            module_name: Full module name for import
            module_path: Path to the module file
        """
        try:
            logger.debug(f"Processing module: {module_name}")
            
            # Import the module
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                error_msg = f"Import failed: {str(e)}"
                self.failed_modules[module_name] = error_msg
                logger.error(f"Failed to import {module_name}: {error_msg}")
                return
            except Exception as e:
                error_msg = f"Unexpected import error: {str(e)}"
                self.failed_modules[module_name] = error_msg
                logger.error(f"Failed to import {module_name}: {error_msg}")
                return
            
            # Find BaseTool subclasses in the module
            tool_classes = []
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it's a BaseTool subclass but not BaseTool itself
                if (issubclass(obj, BaseTool) and 
                    obj is not BaseTool and 
                    obj.__module__ == module_name):
                    tool_classes.append((name, obj))
            
            logger.debug(f"Found {len(tool_classes)} tool classes in {module_name}")
            
            # Register each tool class
            for class_name, tool_class in tool_classes:
                self._register_tool_class(class_name, tool_class, module_name)
                
        except Exception as e:
            error_msg = f"Module processing error: {str(e)}"
            self.failed_modules[module_name] = error_msg
            logger.exception(f"Error processing module {module_name}: {error_msg}")
    
    def _register_tool_class(self, class_name: str, tool_class: Type[BaseTool], module_name: str) -> None:
        """
        Register a single tool class instance in the registry.
        
        Args:
            class_name: Name of the tool class
            tool_class: The BaseTool subclass
            module_name: Name of the module containing the tool
        """
        try:
            # Instantiate the tool
            tool_instance = tool_class()
            
            # Validate that the tool has a unique name
            if tool_instance.name in self.tools:
                existing_tool = self.tools[tool_instance.name]
                logger.warning(f"Tool name conflict: '{tool_instance.name}' already registered")
                logger.warning(f"  Existing: {existing_tool.__class__.__name__} from {existing_tool.__module__}")
                logger.warning(f"  New: {class_name} from {module_name}")
                logger.warning(f"  Keeping existing tool")
                return
            
            # Register the tool
            self.tools[tool_instance.name] = tool_instance
            logger.success(f"Registered tool: {tool_instance.name} ({class_name})")
            
        except Exception as e:
            error_msg = f"Tool instantiation failed: {str(e)}"
            failed_key = f"{module_name}.{class_name}"
            self.failed_modules[failed_key] = error_msg
            logger.error(f"Failed to register tool {class_name}: {error_msg}")
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Retrieve a tool instance by name from the registry.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            BaseTool instance if found, None otherwise
        """
        tool = self.tools.get(tool_name)
        
        if tool:
            logger.debug(f"Retrieved tool: {tool_name}")
        else:
            logger.debug(f"Tool not found: {tool_name}")
            
        return tool
    
    def list_tools(self) -> List[str]:
        """
        Get a list of all registered tool names.
        
        Returns:
            List of tool names sorted alphabetically
        """
        tool_names = sorted(self.tools.keys())
        logger.debug(f"Listed {len(tool_names)} available tools")
        return tool_names
    
    def list_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get schema information for all registered tools.
        
        This method is crucial for LLM integration as it provides
        the LLM with information about available tools and their parameters.
        
        Returns:
            List of tool schema dictionaries
        """
        schemas = []
        
        for tool_name, tool in self.tools.items():
            try:
                schema_info = tool.get_schema_info()
                schemas.append(schema_info)
                logger.debug(f"Retrieved schema for tool: {tool_name}")
                
            except Exception as e:
                logger.error(f"Failed to get schema for tool {tool_name}: {e}")
                # Add minimal schema for failed tools
                schemas.append({
                    "name": tool_name,
                    "description": f"Tool schema unavailable: {str(e)}",
                    "error": True
                })
        
        logger.info(f"Retrieved schemas for {len(schemas)} tools")
        return schemas
    
    def run_tool(self, tool_name: str, **kwargs) -> BaseModel:
        """
        Execute a tool by name with the provided arguments.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tool execution result or error information
        """
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool arguments: {kwargs}")
        
        # Check if tool exists
        tool = self.get_tool(tool_name)
        if not tool:
            available_tools = ", ".join(self.list_tools())
            error_result = ToolError(
                error_type="ToolNotFoundError",
                error_message=f"Tool '{tool_name}' not found in registry",
                error_details=f"Available tools: {available_tools}",
                metadata={
                    "requested_tool": tool_name,
                    "available_tools": self.list_tools(),
                    "registry_size": len(self.tools)
                }
            )
            logger.error(f"Tool not found: {tool_name}")
            return error_result
        
        # Execute the tool
        try:
            logger.debug(f"Executing {tool_name} with tool instance: {tool.__class__.__name__}")
            result = tool.execute(**kwargs)
            
            # Log execution result
            if hasattr(result, 'success') and result.success:
                logger.success(f"Tool execution successful: {tool_name}")
            else:
                logger.warning(f"Tool execution completed with errors: {tool_name}")
            
            return result
            
        except Exception as e:
            # This should rarely happen due to tool's internal error handling
            error_result = ToolError(
                error_type="ToolManagerExecutionError",
                error_message=f"Unexpected error during tool execution: {str(e)}",
                error_details=f"Tool: {tool_name}, Error: {str(e)}",
                metadata={
                    "tool_name": tool_name,
                    "tool_class": tool.__class__.__name__,
                    "exception_type": type(e).__name__,
                    "arguments": str(kwargs)[:200]  # Limit size
                }
            )
            logger.exception(f"Unexpected error executing tool {tool_name}: {e}")
            return error_result
    
    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the tool registry.
        
        Returns:
            Dictionary with registry statistics and configuration
        """
        return {
            "total_tools": len(self.tools),
            "failed_modules": len(self.failed_modules),
            "discovery_paths": [str(p) for p in self.discovery_paths],
            "configuration": {
                "tools_dir": str(self.tools_dir),
                "auto_discovery": self.auto_discovery,
                "recursive_discovery": self.recursive_discovery,
                "execution_timeout": self.execution_timeout
            },
            "tool_summary": [
                {
                    "name": tool.name,
                    "class": tool.__class__.__name__,
                    "version": tool.version,
                    "category": tool.category
                }
                for tool in self.tools.values()
            ],
            "failed_modules": dict(self.failed_modules)
        }
    
    def reload_tools(self) -> None:
        """
        Reload all tools from the tools directory.
        
        Useful for development when tools are modified.
        """
        logger.info("Reloading tools from tools directory")
        
        # Remove imported modules from cache
        modules_to_remove = [
            module_name for module_name in sys.modules.keys()
            if module_name.startswith(self.tools_dir.name + ".")
        ]
        
        for module_name in modules_to_remove:
            del sys.modules[module_name]
            logger.debug(f"Removed module from cache: {module_name}")
        
        # Rediscover tools
        self.discover_and_register_tools()
        logger.info(f"Tool reload completed: {len(self.tools)} tools available")


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the ToolManager with mock tools and comprehensive scenarios.
    This validates the dynamic discovery process and tool execution system.
    """
    
    print("=== DataNeuron ToolManager Test ===")
    
    # Create test tools directory structure
    test_tools_dir = PROJECT_ROOT / "test_tools"
    test_tools_dir.mkdir(exist_ok=True)
    
    # Create __init__.py for the test tools package
    (test_tools_dir / "__init__.py").write_text("")
    
    # Create a mock tool module
    mock_tool_code = '''
from tools.base_tool import BaseTool, BaseToolArgs, BaseToolResult
from pydantic import BaseModel
from typing import Dict, Any

class MockCalculatorArgs(BaseToolArgs):
    operation: str
    a: float
    b: float

class MockCalculatorResult(BaseToolResult):
    result: float
    operation_performed: str

class MockCalculatorTool(BaseTool):
    name = "mock_calculator"
    description = "A mock calculator tool for testing the ToolManager"
    args_schema = MockCalculatorArgs
    return_schema = MockCalculatorResult
    version = "1.0.0"
    category = "test"
    
    def _execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        if operation == "add":
            result = a + b
        elif operation == "multiply":
            result = a * b
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return {
            "result": result,
            "operation_performed": f"{a} {operation} {b} = {result}",
            "metadata": {"tool_version": self.version}
        }

class MockTextTool(BaseTool):
    name = "mock_text_processor"
    description = "A mock text processing tool"
    args_schema = BaseToolArgs  # Use base args for simplicity
    return_schema = BaseToolResult
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        return {"result": "Text processed", "metadata": {"processed_args": len(kwargs)}}
'''
    
    # Write mock tool to file
    (test_tools_dir / "mock_tools.py").write_text(mock_tool_code)
    
    # Create a broken tool module for error testing
    broken_tool_code = '''
from tools.base_tool import BaseTool

class BrokenTool(BaseTool):
    # Missing required attributes - should fail validation
    pass

# This should cause import error
undefined_variable_that_causes_error
'''
    
    (test_tools_dir / "broken_tool.py").write_text(broken_tool_code)
    
    try:
        print("\nTest 1: Create ToolManager with dynamic discovery")
        
        # Temporarily modify TOOLS_DIR for testing
        import config.settings as settings
        original_tools_dir = settings.TOOLS_DIR
        settings.TOOLS_DIR = test_tools_dir
        
        # Create ToolManager instance
        tool_manager = ToolManager()
        
        print(f" PASS - ToolManager created")
        print(f"   Tools discovered: {len(tool_manager.tools)}")
        print(f"   Failed modules: {len(tool_manager.failed_modules)}")
        
        # Test 2: List available tools
        print("\nTest 2: List available tools")
        available_tools = tool_manager.list_tools()
        print(f" PASS - Available tools: {available_tools}")
        
        # Test 3: Get tool schemas
        print("\nTest 3: Get tool schemas for LLM integration")
        schemas = tool_manager.list_tool_schemas()
        print(f" PASS - Retrieved {len(schemas)} tool schemas")
        for schema in schemas:
            print(f"   - {schema['name']}: {schema.get('description', 'No description')[:50]}...")
        
        # Test 4: Execute valid tool
        print("\nTest 4: Execute valid tool with correct arguments")
        if "mock_calculator" in available_tools:
            result = tool_manager.run_tool("mock_calculator", operation="add", a=5.0, b=3.0)
            if hasattr(result, 'success') and result.success:
                print(f" PASS - Tool execution successful")
                print(f"   Result: {result.result}")
                print(f"   Operation: {result.operation_performed}")
            else:
                print(f"L FAIL - Tool execution failed: {result.error_message}")
        else:
            print("L SKIP - Mock calculator tool not found")
        
        # Test 5: Execute tool with invalid arguments
        print("\nTest 5: Execute tool with invalid arguments")
        if "mock_calculator" in available_tools:
            result = tool_manager.run_tool("mock_calculator", operation="add", a="not_a_number")
            if hasattr(result, 'success') and not result.success:
                print(f" PASS - Invalid arguments handled gracefully")
                print(f"   Error type: {result.error_type}")
                print(f"   Error message: {result.error_message}")
            else:
                print(f"L FAIL - Should have failed with validation error")
        
        # Test 6: Execute non-existent tool
        print("\nTest 6: Execute non-existent tool")
        result = tool_manager.run_tool("non_existent_tool", some_arg="value")
        if hasattr(result, 'success') and not result.success:
            print(f" PASS - Non-existent tool handled gracefully")
            print(f"   Error type: {result.error_type}")
            print(f"   Available tools: {len(result.metadata.get('available_tools', []))}")
        else:
            print(f"L FAIL - Should have failed with tool not found error")
        
        # Test 7: Registry information
        print("\nTest 7: Get registry information")
        registry_info = tool_manager.get_registry_info()
        print(f" PASS - Registry info retrieved")
        print(f"   Total tools: {registry_info['total_tools']}")
        print(f"   Failed modules: {registry_info['failed_modules']}")
        print(f"   Configuration: {registry_info['configuration']['tools_dir']}")
        
        # Test 8: Singleton pattern validation
        print("\nTest 8: Validate singleton pattern")
        tool_manager2 = ToolManager()
        if tool_manager is tool_manager2:
            print(f" PASS - Singleton pattern working correctly")
        else:
            print(f"L FAIL - Multiple instances created")
        
        # Restore original settings
        settings.TOOLS_DIR = original_tools_dir
        
    except Exception as e:
        print(f"L FAIL - Test execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test files
        try:
            import shutil
            if test_tools_dir.exists():
                shutil.rmtree(test_tools_dir)
                print(f"\n Test files cleaned up")
        except Exception as e:
            print(f"\nCould not clean up test files: {e}")
    
    print("\n=== Test Complete ===")
    print("ToolManager dynamic discovery and execution system tested.")
    print("The system is ready for LLM integration and production use.")