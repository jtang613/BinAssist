#!/usr/bin/env python3

import asyncio
from PySide6.QtWidgets import QMessageBox, QInputDialog, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox, QComboBox
from PySide6.QtCore import QObject, QThread, Signal
try:
    import binaryninja
    log = binaryninja.log.Logger(0, "BinAssist")
except ImportError:
    # Fallback for testing outside Binary Ninja
    class MockLog:
        @staticmethod
        def log_debug(msg): print(f"[BinAssist] DEBUG: {msg}")
        @staticmethod  
        def log_info(msg): print(f"[BinAssist] INFO: {msg}")
        @staticmethod
        def log_warn(msg): print(f"[BinAssist] WARN: {msg}")
        @staticmethod
        def log_error(msg): print(f"[BinAssist] ERROR: {msg}")
    log = MockLog()
from ..services.settings_service import settings_service
from ..services.models.provider_types import ProviderType
from ..services.mcp_client_service import MCPClientService
from ..services.models.mcp_models import MCPServerConfig, MCPTestResult
from ..services import (
    get_service_registry, LLMService, ChatMessage, MessageRole,
    LLMProviderError, APIProviderError, AuthenticationError, 
    RateLimitError, NetworkError
)
from ..views.settings_tab_view import SettingsTabView


class ProviderTestWorker(QThread):
    """Worker thread for testing provider connectivity"""
    
    # Signals
    test_completed = Signal(bool, str)  # success, message
    
    def __init__(self, provider_config):
        super().__init__()
        self.provider_config = provider_config
    
    def run(self):
        """Run provider test in background thread"""
        provider_name = self.provider_config.get('name', 'Unknown')
        
        try:
            # Log start of test
            log.log_info(f"Testing provider '{provider_name}'...")
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async test with timeout
                success, message = loop.run_until_complete(
                    asyncio.wait_for(self._test_provider(), timeout=30.0)
                )
                
                # Log completion
                if success:
                    log.log_info(f"Provider test successful for '{provider_name}'")
                else:
                    log.log_warn(f"Provider test failed for '{provider_name}'")
                
                self.test_completed.emit(success, message)
                
            except asyncio.TimeoutError:
                log.log_warn(f"Provider test timeout for '{provider_name}' after 30 seconds")
                self.test_completed.emit(False, f"⏱️ Test timeout after 30 seconds")
                
            except Exception as e:
                log.log_error(f"Provider test execution failed for '{provider_name}': {e}")
                self.test_completed.emit(False, f"❌ Test execution error: {str(e)}")
                
            finally:
                # Clean shutdown of event loop
                try:
                    # Cancel any remaining tasks
                    pending = asyncio.all_tasks(loop)
                    if pending:
                        for task in pending:
                            task.cancel()
                        # Wait for cancellation to complete
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception as cleanup_error:
                    log.log_debug(f"Event loop cleanup warning: {cleanup_error}")
                finally:
                    loop.close()
                
        except Exception as e:
            log.log_error(f"Provider test setup failed for '{provider_name}': {e}")
            self.test_completed.emit(False, f"❌ Test setup error: {str(e)}")
    
    async def _test_provider(self):
        """Async method to test provider connectivity"""
        provider_name = self.provider_config.get('name', 'Unknown')
        provider_type = self.provider_config.get('provider_type', 'unknown')
        
        # Store references to avoid issues during cleanup
        registry = None
        llm_service = None
        
        try:
            # Get service registry and initialize
            registry = get_service_registry()
            if not registry.is_initialized():
                registry.initialize()
            
            # Get the LLM service
            llm_service = registry.get_llm_service()
            if not llm_service:
                return False, "❌ LLM service not available"
            
            # Create provider directly for testing (bypass settings service)
            from ..services.llm_providers.provider_factory import get_provider_factory
            
            factory = get_provider_factory()
            test_provider = factory.create_provider(self.provider_config)
            
            if not test_provider:
                return False, f"❌ Failed to create provider '{provider_name}'"
            
            # Create test message
            test_messages = [
                ChatMessage(
                    role=MessageRole.USER, 
                    content="This is a test, please respond with just the word: OK"
                )
            ]
            
            # Perform the test directly with the provider
            log.log_debug(f"Sending test request to {provider_type} provider...")
            
            from ..services.models.llm_models import ChatRequest
            # Use the user's configured settings for testing - don't override them!
            test_request = ChatRequest(
                messages=test_messages,
                model=test_provider.model,
                max_tokens=test_provider.max_tokens,  # Use user's configured max_tokens
                temperature=None  # Let the provider handle temperature based on model type
            )
            
            response = await test_provider.chat_completion(test_request)
            
            # Check if we got a reasonable response
            if response and response.content:
                content = response.content.strip()
                content_upper = content.upper()
                
                log.log_debug(f"Received response: '{content}'")
                
                if "OK" in content_upper:
                    return True, f"✅ Test successful! Response: '{content}'"
                else:
                    return True, f"⚠️ Provider responded but not as expected: '{content}'"
            else:
                log.log_warn(f"Empty or null response from provider")
                return False, "❌ Provider returned empty response"
                
        except AuthenticationError as e:
            log.log_warn(f"Authentication error for {provider_name}: {e}")
            return False, f"🔐 Authentication failed: {str(e)}"
        except RateLimitError as e:
            log.log_warn(f"Rate limit error for {provider_name}: {e}")
            return False, f"⏳ Rate limit exceeded: {str(e)}"
        except NetworkError as e:
            log.log_warn(f"Network error for {provider_name}: {e}")
            return False, f"🌐 Network error: {str(e)}"
        except APIProviderError as e:
            log.log_warn(f"API provider error for {provider_name}: {e}")
            return False, f"🔧 Provider error: {str(e)}"
        except LLMProviderError as e:
            log.log_warn(f"LLM service error for {provider_name}: {e}")
            return False, f"⚡ LLM service error: {str(e)}"
        except Exception as e:
            log.log_error(f"Unexpected error for {provider_name}: {e}")
            return False, f"❌ Unexpected error: {str(e)}"
            
        finally:
            # No cleanup needed since we didn't modify active provider
            pass


class MCPTestWorker(QThread):
    """Worker thread for testing MCP server connectivity"""
    
    # Signals
    test_completed = Signal(bool, str, dict)  # success, message, test_data
    
    def __init__(self, mcp_server_config):
        super().__init__()
        self.server_config = mcp_server_config
    
    def run(self):
        """Run MCP test in background thread"""
        server_name = self.server_config.get('name', 'Unknown')

        try:
            log.log_info(f"Testing MCP server '{server_name}'...")

            # Convert settings format to MCPServerConfig
            transport_type = self.server_config.get('transport', 'sse')

            # Create config based on transport type
            if transport_type == "stdio":
                config = MCPServerConfig(
                    name=self.server_config['name'],
                    transport_type="stdio",
                    command=self.server_config.get('command', ''),
                    enabled=self.server_config.get('enabled', True),
                    timeout=30.0
                )
            else:
                config = MCPServerConfig(
                    name=self.server_config['name'],
                    transport_type="sse",
                    url=self.server_config['url'],
                    enabled=self.server_config.get('enabled', True),
                    timeout=30.0
                )
            
            # Get MCP client service
            mcp_service = MCPClientService()
            
            # Test the connection
            result = mcp_service.test_server_connection(config)
            
            if result.success:
                log.log_info(f"MCP server test successful for '{server_name}'")
                
                # Create detailed result message
                tools_text = f"{result.tools_count} tools" if result.tools_count > 0 else "no tools"
                resources_text = f"{result.resources_count} resources" if result.resources_count > 0 else "no resources"
                
                message = f"✅ Connection successful!\n\nFound {tools_text} and {resources_text}."
                
                if result.tools_count > 0:
                    message += "\n\nAvailable tools:"
                    for i, tool in enumerate(result.tools[:5]):  # Show first 5 tools
                        message += f"\n• {tool['name']}: {tool['description'][:50]}{'...' if len(tool['description']) > 50 else ''}"
                    if result.tools_count > 5:
                        message += f"\n... and {result.tools_count - 5} more tools"
                
                self.test_completed.emit(True, message, result.to_dict())
            else:
                log.log_warn(f"MCP server test failed for '{server_name}'")
                error_message = f"❌ Connection failed: {result.error}"
                
                # Add helpful suggestions for common issues
                if "404" in result.error or "not found" in result.error.lower():
                    error_message += "\n\n💡 Suggestions:"
                    error_message += "\n• Try adding '/sse' to the URL (e.g., http://localhost:9090/sse)"
                    error_message += "\n• Check if the server is running and accessible"
                    error_message += "\n• Verify the server supports MCP over SSE"
                elif "connection refused" in result.error.lower():
                    error_message += "\n\n💡 Suggestion: Check if the server is running on the specified port"
                
                self.test_completed.emit(False, error_message, {})
                
        except Exception as e:
            log.log_error(f"MCP server test execution failed for '{server_name}': {e}")
            error_message = f"❌ Test execution error: {str(e)}"
            self.test_completed.emit(False, error_message, {})


class ProviderDialog(QDialog):
    """Dialog for adding/editing LLM providers"""
    
    def __init__(self, parent=None, provider_data=None):
        super().__init__(parent)
        self.provider_data = provider_data
        self.setup_ui()
        
        if provider_data:
            self.populate_fields()
    
    def setup_ui(self):
        self.setWindowTitle("LLM Provider" if not self.provider_data else f"Edit {self.provider_data.get('name', 'Provider')}")
        self.setModal(True)
        self.resize(400, 350)
        
        layout = QVBoxLayout()
        
        # Name
        layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        layout.addWidget(self.name_edit)
        
        # Provider Type
        layout.addWidget(QLabel("Provider Type:"))
        self.provider_type_combo = QComboBox()
        for provider_type in ProviderType:
            self.provider_type_combo.addItem(provider_type.display_name, provider_type.value)
        
        # Connect signal to update URL and model when provider type changes
        self.provider_type_combo.currentTextChanged.connect(self.on_provider_type_changed)
        layout.addWidget(self.provider_type_combo)
        
        # Model
        layout.addWidget(QLabel("Model:"))
        self.model_edit = QLineEdit()
        layout.addWidget(self.model_edit)
        
        # URL
        layout.addWidget(QLabel("URL:"))
        self.url_edit = QLineEdit()
        layout.addWidget(self.url_edit)
        
        # Max Tokens
        layout.addWidget(QLabel("Max Tokens:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 100000)
        self.max_tokens_spin.setValue(4096)
        layout.addWidget(self.max_tokens_spin)
        
        # API Key
        layout.addWidget(QLabel("API Key:"))
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.key_edit)
        
        # Disable TLS
        self.disable_tls_check = QCheckBox("Disable TLS Verification")
        layout.addWidget(self.disable_tls_check)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def populate_fields(self):
        """Populate fields with existing provider data"""
        if self.provider_data:
            self.name_edit.setText(self.provider_data.get('name', ''))
            
            # Set provider type
            provider_type_value = self.provider_data.get('provider_type', 'openai')
            index = self.provider_type_combo.findData(provider_type_value)
            if index >= 0:
                self.provider_type_combo.setCurrentIndex(index)
            
            self.model_edit.setText(self.provider_data.get('model', ''))
            self.url_edit.setText(self.provider_data.get('url', ''))
            self.max_tokens_spin.setValue(self.provider_data.get('max_tokens', 4096))
            self.key_edit.setText(self.provider_data.get('api_key', ''))
            self.disable_tls_check.setChecked(self.provider_data.get('disable_tls', False))

    def get_provider_data(self):
        """Get the provider data from the form"""
        return {
            'name': self.name_edit.text().strip(),
            'provider_type': self.provider_type_combo.currentData(),
            'model': self.model_edit.text().strip(),
            'url': self.url_edit.text().strip(),
            'max_tokens': self.max_tokens_spin.value(),
            'api_key': self.key_edit.text(),
            'disable_tls': self.disable_tls_check.isChecked()
        }
    
    def on_provider_type_changed(self):
        """Handle provider type selection change"""
        current_data = self.provider_type_combo.currentData()
        if current_data:
            try:
                provider_type = ProviderType(current_data)
                
                # Auto-fill URL with default for this provider type
                if not self.url_edit.text() or self.url_edit.text() in [pt.default_url for pt in ProviderType]:
                    self.url_edit.setText(provider_type.default_url)
                
                # Clear model field to encourage user to select appropriate model
                # Could also populate with default models if desired
                if not self.model_edit.text():
                    # Optionally set first default model
                    default_models = provider_type.default_models
                    if default_models:
                        self.model_edit.setText(default_models[0])
                        
            except ValueError:
                pass  # Invalid provider type, ignore


class MCPProviderDialog(QDialog):
    """Dialog for adding/editing MCP providers"""
    
    def __init__(self, parent=None, provider_data=None):
        super().__init__(parent)
        self.provider_data = provider_data
        self.setup_ui()
        
        if provider_data:
            self.populate_fields()
    
    def setup_ui(self):
        self.setWindowTitle("MCP Provider" if not self.provider_data else f"Edit {self.provider_data.get('name', 'Provider')}")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # Name
        layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        layout.addWidget(self.name_edit)
        
        # URL
        layout.addWidget(QLabel("URL:"))
        self.url_edit = QLineEdit()
        layout.addWidget(self.url_edit)
        
        # Transport
        layout.addWidget(QLabel("Transport:"))
        self.transport_combo = QComboBox()
        self.transport_combo.addItem("SSE (HTTP)", "sse")
        self.transport_combo.addItem("STDIO (Local)", "stdio")
        layout.addWidget(self.transport_combo)
        
        # Command (for STDIO)
        self.command_label = QLabel("Command (for STDIO):")
        layout.addWidget(self.command_label)
        self.command_edit = QLineEdit()
        self.command_edit.setPlaceholderText("e.g., python mcp-server.py")
        layout.addWidget(self.command_edit)
        
        # Show/hide command field based on transport
        self.transport_combo.currentTextChanged.connect(self.on_transport_changed)
        self.on_transport_changed()  # Initial setup
        
        # Enabled
        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.setChecked(True)
        layout.addWidget(self.enabled_check)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def on_transport_changed(self):
        """Handle transport type change"""
        transport_data = self.transport_combo.currentData()
        is_stdio = transport_data == "stdio"
        
        # Show/hide command fields for STDIO
        self.command_label.setVisible(is_stdio)
        self.command_edit.setVisible(is_stdio)
        
        # Update URL placeholder based on transport
        if is_stdio:
            self.url_edit.setPlaceholderText("Leave empty for STDIO transport")
        else:
            self.url_edit.setPlaceholderText("e.g., http://localhost:8000/sse")
    
    def populate_fields(self):
        """Populate fields with existing provider data"""
        if self.provider_data:
            self.name_edit.setText(self.provider_data.get('name', ''))
            self.url_edit.setText(self.provider_data.get('url', ''))

            # Set transport type
            transport_value = self.provider_data.get('transport', 'sse')
            index = self.transport_combo.findData(transport_value)
            if index >= 0:
                self.transport_combo.setCurrentIndex(index)

            self.command_edit.setText(self.provider_data.get('command', ''))
            self.enabled_check.setChecked(self.provider_data.get('enabled', True))

            # Update UI based on transport type
            self.on_transport_changed()
    
    def get_provider_data(self):
        """Get the provider data from the form"""
        data = {
            'name': self.name_edit.text().strip(),
            'url': self.url_edit.text().strip(),
            'transport': self.transport_combo.currentData(),
            'enabled': self.enabled_check.isChecked()
        }

        # Add command for STDIO transport
        if self.transport_combo.currentData() == "stdio":
            data['command'] = self.command_edit.text().strip()

        return data


class SettingsController(QObject):
    """Controller for managing settings between view and service layers"""
    
    def __init__(self, settings_view: SettingsTabView):
        super().__init__()
        self.view = settings_view
        self.service = settings_service
        self.mcp_service = MCPClientService()
        
        # Connect view signals to controller methods
        self.connect_signals()
        
        # Load initial data
        self.load_initial_data()
    
    def connect_signals(self):
        """Connect view signals to controller methods"""
        # LLM Provider signals
        self.view.llm_provider_add_requested.connect(self.add_llm_provider)
        self.view.llm_provider_edit_requested.connect(self.edit_llm_provider)
        self.view.llm_provider_delete_requested.connect(self.delete_llm_provider)
        self.view.llm_provider_test_requested.connect(self.test_llm_provider)
        self.view.llm_active_provider_changed.connect(self.set_active_llm_provider)
        self.view.reasoning_effort_changed.connect(self.update_reasoning_effort)

        # MCP Provider signals
        self.view.mcp_provider_add_requested.connect(self.add_mcp_provider)
        self.view.mcp_provider_edit_requested.connect(self.edit_mcp_provider)
        self.view.mcp_provider_delete_requested.connect(self.delete_mcp_provider)
        self.view.mcp_provider_test_requested.connect(self.test_mcp_provider)
        
        # Settings signals  
        self.view.system_prompt_changed.connect(self.update_system_prompt)
        self.view.database_path_changed.connect(self.update_database_path)
    
    def load_initial_data(self):
        """Load initial data from service into view"""
        try:
            # Disconnect active provider signal during entire load process to prevent premature triggering
            self.view.active_provider_combo.currentTextChanged.disconnect()
            
            # Clear existing data
            self.view.llm_table.setRowCount(0)
            self.view.mcp_table.setRowCount(0)
            self.view.active_provider_combo.clear()
            
            # Load LLM providers
            llm_providers = self.service.get_llm_providers()
            for provider in llm_providers:
                self.view.add_llm_provider(
                    provider['name'],
                    provider['model'],
                    provider.get('provider_type', 'openai'),
                    provider['url'],
                    provider['max_tokens'],
                    provider['api_key'],
                    provider['disable_tls']
                )
            
            # Load MCP providers
            mcp_providers = self.service.get_mcp_providers()
            for provider in mcp_providers:
                self.view.add_mcp_provider(
                    provider['name'],
                    provider['url'],
                    provider['enabled'],
                    provider['transport']
                )
            
            # Set active provider (signal already disconnected at top level)
            active_provider = self.service.get_active_llm_provider()
            log.log_debug(f"Loading active provider: {active_provider}")
            if active_provider and self.view.active_provider_combo.count() > 0:
                provider_name = active_provider['name']
                index = self.view.active_provider_combo.findText(provider_name)
                log.log_debug(f"Looking for provider '{provider_name}', found at index: {index}")
                if index >= 0:
                    self.view.active_provider_combo.setCurrentIndex(index)
                    log.log_debug(f"Set active provider combo to index {index} ({provider_name})")

                    # Set reasoning effort combo to match active provider
                    reasoning_effort = active_provider.get('reasoning_effort', 'none')
                    self.view.set_reasoning_effort(reasoning_effort)
                    log.log_debug(f"Set reasoning effort to: {reasoning_effort}")
                else:
                    log.log_warn(f"Provider '{provider_name}' not found in combo box")
                    # List available providers for debugging
                    available = [self.view.active_provider_combo.itemText(i) for i in range(self.view.active_provider_combo.count())]
                    log.log_warn(f"Available providers: {available}")
            else:
                log.log_debug(f"No active provider to restore or combo box is empty (count: {self.view.active_provider_combo.count()})")
            
            # Load system prompt
            system_prompt = self.service.get_setting('system_prompt', '')
            self.view.system_prompt_text.setPlainText(system_prompt)
            
            # Load database paths
            self.view.analysis_db_path.setText(self.service.get_setting('analysis_db_path', ''))
            self.view.rlhf_db_path.setText(self.service.get_setting('rlhf_db_path', ''))
            self.view.rag_index_path.setText(self.service.get_setting('rag_index_path', ''))
            
        except Exception as e:
            self.show_error("Failed to load settings", str(e))
        finally:
            # Always reconnect the signal after loading is complete
            self.view.active_provider_combo.currentTextChanged.connect(self.set_active_llm_provider)
    
    # LLM Provider methods
    
    def add_llm_provider(self):
        """Handle adding a new LLM provider"""
        dialog = ProviderDialog(self.view)
        if dialog.exec() == QDialog.Accepted:
            try:
                data = dialog.get_provider_data()
                
                # Validate required fields
                if not all([data['name'], data['model'], data['url']]):
                    self.show_error("Validation Error", "Name, Model, and URL are required fields.")
                    return
                
                # Add to service
                provider_id = self.service.add_llm_provider(
                    data['name'], data['model'], data['url'],
                    data['max_tokens'], data['api_key'], data['disable_tls'], data['provider_type']
                )
                
                # If this is the first provider, set it as active
                providers = self.service.get_llm_providers()
                if len(providers) == 1:
                    self.service.set_active_llm_provider(data['name'])
                
                # Refresh the entire view to avoid race conditions
                self.load_initial_data()
                
                self.show_info("Success", f"Added LLM provider '{data['name']}'")
                
            except ValueError as e:
                self.show_error("Provider Already Exists", str(e))
            except Exception as e:
                self.show_error("Failed to Add Provider", str(e))
    
    def edit_llm_provider(self, row):
        """Handle editing an LLM provider"""
        if row < 0 or row >= self.view.llm_table.rowCount():
            return
        
        try:
            # Get current provider data
            providers = self.service.get_llm_providers()
            if row >= len(providers):
                return
            
            provider = providers[row]
            dialog = ProviderDialog(self.view, provider)
            
            if dialog.exec() == QDialog.Accepted:
                data = dialog.get_provider_data()
                
                # Validate required fields
                if not all([data['name'], data['model'], data['url']]):
                    self.show_error("Validation Error", "Name, Model, and URL are required fields.")
                    return
                
                # Update in service
                self.service.update_llm_provider(provider['id'], **data)
                
                # Reload data to refresh view
                self.load_initial_data()
                
                self.show_info("Success", f"Updated LLM provider '{data['name']}'")
                
        except Exception as e:
            self.show_error("Failed to Update Provider", str(e))
    
    def delete_llm_provider(self, row):
        """Handle deleting an LLM provider"""
        if row < 0 or row >= self.view.llm_table.rowCount():
            return
        
        try:
            providers = self.service.get_llm_providers()
            if row >= len(providers):
                return
            
            provider = providers[row]
            
            # Confirm deletion
            reply = QMessageBox.question(
                self.view, "Confirm Deletion",
                f"Are you sure you want to delete the LLM provider '{provider['name']}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Delete from service
                self.service.delete_llm_provider(provider['id'])
                
                # Remove from view
                self.view.llm_table.removeRow(row)
                self.view.active_provider_combo.removeItem(row)
                
                self.show_info("Success", f"Deleted LLM provider '{provider['name']}'")
                
        except Exception as e:
            self.show_error("Failed to Delete Provider", str(e))
    
    def test_llm_provider(self, row):
        """Handle testing an LLM provider"""
        try:
            providers = self.service.get_llm_providers()
            if row >= len(providers):
                return
            
            provider = providers[row]
            
            # Validate that provider has required fields
            if not provider.get('api_key') and provider.get('provider_type') != 'ollama':
                self.show_error("Test Failed", 
                    f"Provider '{provider['name']}' has no API key configured. "
                    "Please edit the provider and add an API key before testing.")
                return
            
            # Disable the test button to prevent multiple concurrent tests
            self.view.llm_test_button.setEnabled(False)
            self.view.llm_test_button.setText("Testing...")
            
            # Create and start the test worker
            self.test_worker = ProviderTestWorker(provider)
            self.test_worker.test_completed.connect(self.on_provider_test_completed)
            self.test_worker.finished.connect(self.test_worker.deleteLater)
            self.test_worker.start()
            
        except Exception as e:
            self.show_error("Test Failed", str(e))
            # Re-enable button on error
            self.view.llm_test_button.setEnabled(True)
            self.view.llm_test_button.setText("Test")
    
    def on_provider_test_completed(self, success, message):
        """Handle provider test completion"""
        # Re-enable the test button
        self.view.llm_test_button.setEnabled(True)
        self.view.llm_test_button.setText("Test")
        
        # Show result to user
        if success:
            self.show_info("Provider Test Result", message)
        else:
            self.show_error("Provider Test Failed", message)
    
    def set_active_llm_provider(self, provider_name):
        """Handle setting active LLM provider"""
        log.log_debug(f"set_active_llm_provider called with: '{provider_name}'")
        if not provider_name:
            log.log_debug("Empty provider name, returning")
            return

        try:
            # Temporarily disconnect the signal to avoid recursive calls
            self.view.active_provider_combo.currentTextChanged.disconnect()

            success = self.service.set_active_llm_provider(provider_name)
            log.log_debug(f"set_active_llm_provider result: {success}")

            # Update reasoning effort combo to match new active provider
            provider = self.service.get_active_llm_provider()
            if provider:
                reasoning_effort = provider.get('reasoning_effort', 'none')
                self.view.set_reasoning_effort(reasoning_effort)

            # Reconnect the signal
            self.view.active_provider_combo.currentTextChanged.connect(self.set_active_llm_provider)

        except Exception as e:
            log.log_error(f"Error setting active provider: {e}")
            # Reconnect signal even if there was an error
            try:
                self.view.active_provider_combo.currentTextChanged.connect(self.set_active_llm_provider)
            except:
                pass
            self.show_error("Failed to Set Active Provider", str(e))

    def update_reasoning_effort(self, reasoning_effort: str):
        """Handle reasoning effort change for active provider"""
        log.log_debug(f"update_reasoning_effort called with: '{reasoning_effort}'")
        try:
            # Get active provider
            provider = self.service.get_active_llm_provider()
            if not provider:
                log.log_warn("No active provider to update reasoning effort")
                return

            provider_id = provider.get('id')
            if provider_id:
                # Update the reasoning effort for the active provider
                self.service.update_llm_provider(provider_id, reasoning_effort=reasoning_effort)
                log.log_info(f"Updated reasoning effort to '{reasoning_effort}' for provider: {provider.get('name')}")
            else:
                log.log_error("Active provider has no ID")

        except Exception as e:
            log.log_error(f"Error updating reasoning effort: {e}")
            self.show_error("Failed to Update Reasoning Effort", str(e))
    
    # MCP Provider methods
    
    def add_mcp_provider(self):
        """Handle adding a new MCP provider"""
        dialog = MCPProviderDialog(self.view)
        if dialog.exec() == QDialog.Accepted:
            try:
                data = dialog.get_provider_data()
                
                # Validate required fields
                if not all([data['name'], data['url']]):
                    self.show_error("Validation Error", "Name and URL are required fields.")
                    return
                
                # Add to service
                provider_id = self.service.add_mcp_provider(
                    data['name'], data['url'], data['enabled'], data['transport']
                )
                
                # Refresh the entire view to avoid race conditions  
                self.load_initial_data()
                
                self.show_info("Success", f"Added MCP provider '{data['name']}'")
                
            except ValueError as e:
                self.show_error("Provider Already Exists", str(e))
            except Exception as e:
                self.show_error("Failed to Add Provider", str(e))
    
    def edit_mcp_provider(self, row):
        """Handle editing an MCP provider"""
        if row < 0 or row >= self.view.mcp_table.rowCount():
            return
        
        try:
            providers = self.service.get_mcp_providers()
            if row >= len(providers):
                return
            
            provider = providers[row]
            dialog = MCPProviderDialog(self.view, provider)
            
            if dialog.exec() == QDialog.Accepted:
                data = dialog.get_provider_data()
                
                # Validate required fields
                if not all([data['name'], data['url']]):
                    self.show_error("Validation Error", "Name and URL are required fields.")
                    return
                
                # Update in service
                self.service.update_mcp_provider(provider['id'], **data)
                
                # Reload data to refresh view
                self.load_initial_data()
                
                self.show_info("Success", f"Updated MCP provider '{data['name']}'")
                
        except Exception as e:
            self.show_error("Failed to Update Provider", str(e))
    
    def delete_mcp_provider(self, row):
        """Handle deleting an MCP provider"""
        if row < 0 or row >= self.view.mcp_table.rowCount():
            return
        
        try:
            providers = self.service.get_mcp_providers()
            if row >= len(providers):
                return
            
            provider = providers[row]
            
            # Confirm deletion
            reply = QMessageBox.question(
                self.view, "Confirm Deletion",
                f"Are you sure you want to delete the MCP provider '{provider['name']}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Delete from service
                self.service.delete_mcp_provider(provider['id'])
                
                # Remove from view
                self.view.mcp_table.removeRow(row)
                
                self.show_info("Success", f"Deleted MCP provider '{provider['name']}'")
                
        except Exception as e:
            self.show_error("Failed to Delete Provider", str(e))
    
    def test_mcp_provider(self, row):
        """Handle testing an MCP provider"""
        try:
            providers = self.service.get_mcp_providers()
            if row >= len(providers):
                return
            
            provider = providers[row]
            
            # Validate that provider has required fields
            if not provider.get('url'):
                self.show_error("Test Failed", 
                    f"Provider '{provider['name']}' has no URL configured. "
                    "Please edit the provider and add a URL before testing.")
                return
            
            # Disable the test button to prevent multiple concurrent tests
            self.view.mcp_test_button.setEnabled(False)
            self.view.mcp_test_button.setText("Testing...")
            
            # Create and start the MCP test worker
            self.mcp_test_worker = MCPTestWorker(provider)
            self.mcp_test_worker.test_completed.connect(self.on_mcp_test_completed)
            self.mcp_test_worker.finished.connect(self.mcp_test_worker.deleteLater)
            self.mcp_test_worker.start()
            
        except Exception as e:
            self.show_error("Test Failed", str(e))
            # Re-enable button on error
            self.view.mcp_test_button.setEnabled(True)
            self.view.mcp_test_button.setText("Test")
    
    def on_mcp_test_completed(self, success, message, test_data):
        """Handle MCP test completion"""
        # Re-enable the test button
        self.view.mcp_test_button.setEnabled(True)
        self.view.mcp_test_button.setText("Test")
        
        # Show result to user
        if success:
            self.show_info("MCP Server Test Result", message)
        else:
            self.show_error("MCP Server Test Failed", message)
    
    # Settings methods
    
    def update_system_prompt(self, prompt_text):
        """Handle system prompt updates"""
        try:
            self.service.set_setting('system_prompt', prompt_text, 'system')
        except Exception as e:
            self.show_error("Failed to Update System Prompt", str(e))
    
    def update_database_path(self, path_type, path_value):
        """Handle database path updates"""
        try:
            setting_key = f"{path_type}_path"
            self.service.set_setting(setting_key, path_value, 'database')
        except Exception as e:
            self.show_error("Failed to Update Database Path", str(e))
    
    # Utility methods
    
    def show_error(self, title, message):
        """Show error message dialog"""
        QMessageBox.critical(self.view, title, message)
    
    def show_info(self, title, message):
        """Show info message dialog"""
        QMessageBox.information(self.view, title, message)