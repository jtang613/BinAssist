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

            # Create config for HTTP-based transport (sse or streamablehttp)
            config = MCPServerConfig(
                name=self.server_config['name'],
                transport_type=transport_type,
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
        self._pending_pkce_verifier = None  # For OAuth PKCE flow
        self.setup_ui()
        
        if provider_data:
            self.populate_fields()
    
    def setup_ui(self):
        self.setWindowTitle("LLM Provider" if not self.provider_data else f"Edit {self.provider_data.get('name', 'Provider')}")
        self.setModal(True)
        self.resize(450, 400)
        
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
        
        # API Key / OAuth Token (dynamic label)
        self.api_key_label = QLabel("API Key:")
        layout.addWidget(self.api_key_label)
        
        # API Key field with optional Authenticate button
        key_layout = QHBoxLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        key_layout.addWidget(self.key_edit)
        
        # Authenticate button (only visible for OAuth providers)
        self.authenticate_button = QPushButton("Authenticate")
        self.authenticate_button.setToolTip("Open browser to authenticate with Claude Pro/Max")
        self.authenticate_button.clicked.connect(self.on_authenticate_clicked)
        self.authenticate_button.setVisible(False)
        key_layout.addWidget(self.authenticate_button)
        
        layout.addLayout(key_layout)
        
        # OAuth note (only visible for OAuth providers)
        self.oauth_note_label = QLabel(
            "Click 'Authenticate' to sign in with your Claude Pro/Max subscription.\n"
            "The OAuth token will be stored as JSON in the field above."
        )
        self.oauth_note_label.setStyleSheet("color: #666; font-style: italic;")
        self.oauth_note_label.setWordWrap(True)
        layout.addWidget(self.oauth_note_label)
        self.oauth_note_label.setVisible(False)
        
        # Disable TLS
        self.disable_tls_check = QCheckBox("Disable TLS Verification")
        layout.addWidget(self.disable_tls_check)

        # Claude Code CLI note (only visible for Claude Code providers)
        self.claude_code_note_label = QLabel(
            "Note: Requires `claude` CLI installed and authenticated.\n"
            "Install with: npm install -g @anthropic-ai/claude-code"
        )
        self.claude_code_note_label.setStyleSheet("color: #666; font-style: italic; margin-top: 10px;")
        self.claude_code_note_label.setWordWrap(True)
        layout.addWidget(self.claude_code_note_label)
        self.claude_code_note_label.setVisible(False)

        # LiteLLM metadata (read-only fields, only visible for LiteLLM providers)
        self.litellm_metadata_label = QLabel("LiteLLM Metadata:")
        self.litellm_metadata_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(self.litellm_metadata_label)

        self.model_family_label = QLabel("Model Family: Not detected")
        layout.addWidget(self.model_family_label)

        self.is_bedrock_label = QLabel("Bedrock Model: No")
        layout.addWidget(self.is_bedrock_label)

        # Initially hide LiteLLM metadata
        self.litellm_metadata_label.setVisible(False)
        self.model_family_label.setVisible(False)
        self.is_bedrock_label.setVisible(False)

        # Connect model edit to update metadata
        self.model_edit.textChanged.connect(self.update_litellm_metadata)

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
            provider_type_value = self.provider_data.get('provider_type', 'openai_platform')
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

                # Show/hide LiteLLM metadata based on provider type
                is_litellm = provider_type == ProviderType.LITELLM
                self.litellm_metadata_label.setVisible(is_litellm)
                self.model_family_label.setVisible(is_litellm)
                self.is_bedrock_label.setVisible(is_litellm)

                # Update metadata if switching to LiteLLM
                if is_litellm:
                    self.update_litellm_metadata()

                # Show/hide Claude Code note based on provider type
                is_claude_code = provider_type == ProviderType.ANTHROPIC_CLI
                self.claude_code_note_label.setVisible(is_claude_code)

                # Show/hide OAuth elements based on provider type
                is_anthropic_oauth = provider_type == ProviderType.ANTHROPIC_OAUTH
                is_codex_oauth = provider_type == ProviderType.OPENAI_OAUTH
                is_oauth = is_anthropic_oauth or is_codex_oauth
                self.authenticate_button.setVisible(is_oauth)
                self.oauth_note_label.setVisible(is_oauth)
                
                # Update API key label, echo mode, and note text for OAuth
                if is_anthropic_oauth:
                    self.api_key_label.setText("OAuth Token (JSON):")
                    self.key_edit.setEchoMode(QLineEdit.Normal)  # Show JSON for OAuth
                    self.key_edit.setPlaceholderText('Click "Authenticate" to sign in')
                    self.oauth_note_label.setText(
                        "Click 'Authenticate' to sign in with your Claude Pro/Max subscription.\n"
                        "The OAuth token will be stored as JSON in the field above."
                    )
                elif is_codex_oauth:
                    self.api_key_label.setText("OAuth Token (JSON):")
                    self.key_edit.setEchoMode(QLineEdit.Normal)  # Show JSON for OAuth
                    self.key_edit.setPlaceholderText('Click "Authenticate" to sign in with ChatGPT')
                    self.oauth_note_label.setText(
                        "Click 'Authenticate' to sign in with your ChatGPT Pro/Plus subscription.\n"
                        "After authorization, copy the code from the browser URL bar."
                    )
                else:
                    self.api_key_label.setText("API Key:")
                    self.key_edit.setEchoMode(QLineEdit.Password)  # Hide API keys
                    self.key_edit.setPlaceholderText('')

            except ValueError:
                pass  # Invalid provider type, ignore

    def update_litellm_metadata(self):
        """Update LiteLLM metadata labels based on current model name"""
        current_provider_type = self.provider_type_combo.currentData()
        if current_provider_type != 'litellm':
            return

        model_name = self.model_edit.text().strip()
        if not model_name:
            self.model_family_label.setText("Model Family: Not detected")
            self.is_bedrock_label.setText("Bedrock Model: No")
            return

        # Detect model family (same logic as in settings_service.py)
        model_family = self._detect_model_family(model_name)
        is_bedrock = model_name.lower().startswith('bedrock/') or model_name.lower().startswith('bedrock-')

        self.model_family_label.setText(f"Model Family: {model_family.title()}")
        self.is_bedrock_label.setText(f"Bedrock Model: {'Yes' if is_bedrock else 'No'}")

    def _detect_model_family(self, model: str) -> str:
        """Detect model family from model name (same logic as settings_service.py)"""
        model_lower = model.lower()

        # Bedrock models
        if model_lower.startswith('bedrock/'):
            if 'anthropic' in model_lower or 'claude' in model_lower:
                return 'anthropic'
            elif 'amazon' in model_lower or 'nova' in model_lower:
                return 'amazon'
            elif 'meta' in model_lower or 'llama' in model_lower:
                return 'meta'
            elif 'cohere' in model_lower:
                return 'cohere'
            elif 'ai21' in model_lower:
                return 'ai21'

        # Non-Bedrock
        if 'claude' in model_lower or 'anthropic' in model_lower:
            return 'anthropic'
        elif 'gpt' in model_lower or 'openai' in model_lower:
            return 'openai'
        elif 'gemini' in model_lower or 'google' in model_lower:
            return 'google'
        elif 'llama' in model_lower or 'meta' in model_lower:
            return 'meta'

        return 'unknown'

    def on_authenticate_clicked(self):
        """Handle OAuth authentication button click"""
        from PySide6.QtWidgets import QMessageBox, QInputDialog
        import asyncio
        
        # Determine which OAuth provider is selected
        current_data = self.provider_type_combo.currentData()
        try:
            provider_type = ProviderType(current_data)
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid provider type selected.")
            return
        
        is_codex = provider_type == ProviderType.OPENAI_OAUTH
        is_anthropic = provider_type == ProviderType.ANTHROPIC_OAUTH
        
        if is_codex:
            self._authenticate_codex()
        elif is_anthropic:
            self._authenticate_anthropic()
        else:
            QMessageBox.warning(self, "Not Supported", 
                "OAuth authentication is not supported for this provider type.")
    
    def _authenticate_anthropic(self):
        """Handle Anthropic (Claude) OAuth authentication with automatic callback"""
        from PySide6.QtWidgets import QMessageBox, QProgressDialog
        from PySide6.QtCore import Qt
        
        try:
            from ..services.llm_providers.oauth_worker import AnthropicOAuthWorker
            
            log.log_info("Starting Anthropic OAuth authentication with automatic callback")
            
            # Track authentication state to prevent race condition with cancel handler
            self._oauth_auth_completed = False
            
            # Show progress dialog
            self._oauth_progress = QProgressDialog(
                "Waiting for authentication...\n\n"
                "Please complete sign-in in your browser.\n"
                "This dialog will close automatically when done.",
                "Use Manual Entry",
                0, 0,  # Indeterminate progress
                self
            )
            self._oauth_progress.setWindowTitle("Claude Authentication")
            self._oauth_progress.setWindowModality(Qt.WindowModal)
            self._oauth_progress.setMinimumDuration(0)
            self._oauth_progress.setMinimumWidth(400)
            
            # Create and configure worker
            self._oauth_worker = AnthropicOAuthWorker(timeout=300)
            self._oauth_worker.authentication_complete.connect(self._on_anthropic_auth_complete)
            self._oauth_worker.authentication_failed.connect(self._on_anthropic_auth_failed)
            self._oauth_worker.status_update.connect(
                lambda msg: self._oauth_progress.setLabelText(msg) if self._oauth_progress else None
            )
            
            # Handle cancel button - request cancellation, let worker signal handle the rest
            def on_cancel():
                # Check if auth already completed successfully
                if getattr(self, '_oauth_auth_completed', False):
                    return  # Auth succeeded, don't show manual dialog
                
                # Request cancellation - the worker will emit authentication_failed("cancelled")
                # which will trigger _on_anthropic_auth_failed to show manual dialog
                if hasattr(self, '_oauth_worker') and self._oauth_worker:
                    self._oauth_worker.cancel()
            
            self._oauth_progress.canceled.connect(on_cancel)
            
            # Clean up worker when done
            self._oauth_worker.finished.connect(self._oauth_worker.deleteLater)
            
            # Start the worker and show dialog
            self._oauth_worker.start()
            self._oauth_progress.show()
            
        except ImportError as e:
            log.log_error(f"OAuth import error: {e}")
            self._authenticate_anthropic_manual()
        except Exception as e:
            log.log_error(f"OAuth authentication error: {e}")
            self._authenticate_anthropic_manual()
    
    def _on_anthropic_auth_complete(self, result: dict):
        """Handle successful Anthropic OAuth authentication"""
        from PySide6.QtWidgets import QMessageBox
        
        # Mark authentication as completed to prevent cancel handler from firing
        self._oauth_auth_completed = True
        
        # Close progress dialog (this may trigger canceled signal, but we've set the flag)
        if hasattr(self, '_oauth_progress') and self._oauth_progress:
            self._oauth_progress.close()
            self._oauth_progress = None
        
        # Store credentials
        credentials_json = result.get('credentials_json', '')
        self.key_edit.setText(credentials_json)
        
        log.log_info("OAuth authentication successful - token stored in API key field")
        QMessageBox.information(self, "Authentication Successful", 
            "Successfully authenticated with Claude Pro/Max!\n\n"
            "The OAuth token has been stored. Click OK to save the provider.")
    
    def _on_anthropic_auth_failed(self, error: str):
        """Handle failed Anthropic OAuth authentication"""
        from PySide6.QtWidgets import QMessageBox
        
        # Mark as completed (even though failed) to prevent cancel handler race
        self._oauth_auth_completed = True
        
        # Close progress dialog
        if hasattr(self, '_oauth_progress') and self._oauth_progress:
            self._oauth_progress.close()
            self._oauth_progress = None
        
        # Handle different error types - Anthropic likely rejects localhost, so always offer manual
        if error == "cancelled":
            # User cancelled - show manual entry dialog
            log.log_info("User cancelled automatic OAuth, showing manual entry")
            self._authenticate_anthropic_manual()
            return
        elif error == "timeout":
            reply = QMessageBox.question(
                self,
                "Authentication Timeout",
                "The authentication request timed out.\n\n"
                "Would you like to try manual code entry instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._authenticate_anthropic_manual()
        else:
            # For any error (including redirect rejection), fall back to manual
            log.log_info(f"Anthropic OAuth failed ({error}), falling back to manual entry")
            self._authenticate_anthropic_manual()
    
    def _authenticate_anthropic_manual(self):
        """Manual fallback for Anthropic OAuth authentication using hosted callback"""
        from PySide6.QtWidgets import QMessageBox, QInputDialog
        import asyncio
        
        try:
            from ..services.llm_providers.oauth_utils import (
                generate_pkce,
                generate_state,
                open_auth_browser,
                exchange_code_for_tokens,
                create_credentials_json
            )
            
            # Generate PKCE codes and state
            verifier, challenge = generate_pkce()
            state = generate_state()
            self._pending_pkce_verifier = verifier
            self._pending_oauth_state = state
            
            # Open browser for authorization (uses Anthropic's hosted callback)
            auth_url = open_auth_browser(challenge, state)
            log.log_info(f"Opened browser for Anthropic OAuth authorization (manual mode)")
            
            # Show dialog to get authorization code from user
            # Note: For manual flow, the user will get code#state format from the hosted callback
            code, ok = QInputDialog.getText(
                self,
                "OAuth Authentication (Manual)",
                "A browser window has been opened for Claude Pro/Max authentication.\n\n"
                "1. Sign in to your Anthropic account in the browser\n"
                "2. Authorize BinAssist to access your account\n"
                "3. Copy the authorization code shown in the browser\n"
                "   (Format: code#state or just the code)\n"
                "4. Paste it below:\n\n"
                "Authorization Code:"
            )
            
            if not ok or not code.strip():
                log.log_info("OAuth authentication cancelled by user")
                QMessageBox.information(self, "Authentication Cancelled", 
                    "Authentication was cancelled.")
                return
            
            # Parse code - may be in format "code#state" from hosted callback
            code_input = code.strip()
            if "#" in code_input:
                auth_code, returned_state = code_input.split("#", 1)
            else:
                auth_code = code_input
                returned_state = state  # Use original state if not in input
            
            # Exchange code for tokens
            log.log_info("Exchanging authorization code for tokens...")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                tokens = loop.run_until_complete(
                    exchange_code_for_tokens(auth_code, verifier, returned_state)
                )
            finally:
                loop.close()
            
            if tokens.get("error"):
                error_msg = tokens.get('error_description', tokens['error'])
                log.log_error(f"OAuth token exchange failed: {error_msg}")
                QMessageBox.critical(self, "Authentication Failed", 
                    f"Token exchange failed: {error_msg}")
                return
            
            # Create JSON credentials and store
            credentials_json = create_credentials_json(tokens)
            self.key_edit.setText(credentials_json)
            
            log.log_info("OAuth authentication successful - token stored in API key field")
            QMessageBox.information(self, "Authentication Successful", 
                "Successfully authenticated with Claude Pro/Max!\n\n"
                "The OAuth token has been stored. Click OK to save the provider.")
            
        except ImportError as e:
            log.log_error(f"OAuth import error: {e}")
            QMessageBox.critical(self, "Missing Dependencies", 
                f"OAuth authentication requires additional packages.\n\n"
                f"Please install: pip install aiohttp\n\n"
                f"Error: {e}")
        except Exception as e:
            log.log_error(f"OAuth authentication error: {e}")
            QMessageBox.critical(self, "Authentication Error", 
                f"An error occurred during authentication:\n\n{e}")
    
    def _authenticate_codex(self):
        """Handle OpenAI Codex (ChatGPT) OAuth authentication with automatic callback"""
        from PySide6.QtWidgets import QMessageBox, QProgressDialog
        from PySide6.QtCore import Qt
        
        try:
            from ..services.llm_providers.oauth_worker import OpenAIOAuthWorker
            
            log.log_info("Starting OpenAI OAuth authentication with automatic callback")
            
            # Track authentication state to prevent race condition with cancel handler
            self._oauth_auth_completed = False
            
            # Show progress dialog
            self._oauth_progress = QProgressDialog(
                "Waiting for authentication...\n\n"
                "Please complete sign-in in your browser.\n"
                "This dialog will close automatically when done.",
                "Use Manual Entry",
                0, 0,  # Indeterminate progress
                self
            )
            self._oauth_progress.setWindowTitle("OpenAI Authentication")
            self._oauth_progress.setWindowModality(Qt.WindowModal)
            self._oauth_progress.setMinimumDuration(0)
            self._oauth_progress.setMinimumWidth(400)
            
            # Create and configure worker
            self._oauth_worker = OpenAIOAuthWorker(timeout=300)
            self._oauth_worker.authentication_complete.connect(self._on_codex_auth_complete)
            self._oauth_worker.authentication_failed.connect(self._on_codex_auth_failed)
            self._oauth_worker.status_update.connect(
                lambda msg: self._oauth_progress.setLabelText(msg) if self._oauth_progress else None
            )
            
            # Handle cancel button - request cancellation, let worker signal handle the rest
            def on_cancel():
                # Check if auth already completed successfully
                if getattr(self, '_oauth_auth_completed', False):
                    return  # Auth succeeded, don't show manual dialog
                
                # Request cancellation - the worker will emit authentication_failed("cancelled")
                # which will trigger _on_codex_auth_failed to show manual dialog
                if hasattr(self, '_oauth_worker') and self._oauth_worker:
                    self._oauth_worker.cancel()
            
            self._oauth_progress.canceled.connect(on_cancel)
            
            # Clean up worker when done
            self._oauth_worker.finished.connect(self._oauth_worker.deleteLater)
            
            # Start the worker and show dialog
            self._oauth_worker.start()
            self._oauth_progress.show()
            
        except ImportError as e:
            log.log_error(f"OAuth import error: {e}")
            self._authenticate_codex_manual()
        except Exception as e:
            log.log_error(f"OAuth authentication error: {e}")
            self._authenticate_codex_manual()
    
    def _on_codex_auth_complete(self, result: dict):
        """Handle successful OpenAI OAuth authentication"""
        from PySide6.QtWidgets import QMessageBox
        
        # Mark authentication as completed to prevent cancel handler from firing
        self._oauth_auth_completed = True
        
        # Close progress dialog (this may trigger canceled signal, but we've set the flag)
        if hasattr(self, '_oauth_progress') and self._oauth_progress:
            self._oauth_progress.close()
            self._oauth_progress = None
        
        # Store credentials
        credentials_json = result.get('credentials_json', '')
        self.key_edit.setText(credentials_json)
        
        log.log_info("OAuth authentication successful - token stored in API key field")
        QMessageBox.information(self, "Authentication Successful", 
            "Successfully authenticated with ChatGPT Pro/Plus!\n\n"
            "The OAuth token has been stored. Click OK to save the provider.")
    
    def _on_codex_auth_failed(self, error: str):
        """Handle failed OpenAI OAuth authentication"""
        from PySide6.QtWidgets import QMessageBox
        
        # Mark as completed (even though failed) to prevent cancel handler race
        self._oauth_auth_completed = True
        
        # Close progress dialog
        if hasattr(self, '_oauth_progress') and self._oauth_progress:
            self._oauth_progress.close()
            self._oauth_progress = None
        
        # Handle different error types
        if error == "cancelled":
            # User cancelled - show manual entry dialog
            log.log_info("User cancelled automatic OAuth, showing manual entry")
            self._authenticate_codex_manual()
            return
        elif error == "timeout":
            reply = QMessageBox.question(
                self,
                "Authentication Timeout",
                "The authentication request timed out.\n\n"
                "Would you like to try manual code entry instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._authenticate_codex_manual()
        elif error.startswith("server_error:"):
            log.log_info("Falling back to manual entry due to server error")
            self._authenticate_codex_manual()
        else:
            # For other errors, offer manual fallback
            reply = QMessageBox.question(
                self,
                "Authentication Error",
                f"An error occurred during authentication:\n\n{error}\n\n"
                "Would you like to try manual code entry instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._authenticate_codex_manual()
    
    def _authenticate_codex_manual(self):
        """Manual fallback for OpenAI Codex OAuth authentication"""
        from PySide6.QtWidgets import QMessageBox, QInputDialog
        import asyncio
        
        try:
            from ..services.llm_providers.oauth_codex_utils import (
                generate_pkce,
                generate_state,
                open_auth_browser,
                exchange_code_for_tokens,
                create_credentials_json,
                extract_account_id
            )
            
            # Generate PKCE codes and state
            verifier, challenge = generate_pkce()
            state = generate_state()
            self._pending_pkce_verifier = verifier
            
            # Open browser for authorization
            auth_url = open_auth_browser(challenge, state)
            log.log_info(f"Opened browser for OpenAI Codex OAuth authorization (manual mode)")
            
            # Show dialog to get authorization code from user
            code, ok = QInputDialog.getText(
                self,
                "OAuth Authentication (Manual)",
                "A browser window has been opened for ChatGPT Pro/Plus authentication.\n\n"
                "1. Sign in to your OpenAI/ChatGPT account in the browser\n"
                "2. Authorize BinAssist to access your account\n"
                "3. After authorization, the browser will show an error page\n"
                "   (This is expected - the redirect URL won't load)\n"
                "4. Look at the browser's URL bar - it will look like:\n"
                "   http://localhost:1455/auth/callback?code=XXXX&state=YYYY\n"
                "5. Copy the ENTIRE URL or just the 'code' value\n"
                "6. Paste it below:\n\n"
                "Authorization Code or Full URL:"
            )
            
            if not ok or not code.strip():
                log.log_info("OAuth authentication cancelled by user")
                QMessageBox.information(self, "Authentication Cancelled", 
                    "Authentication was cancelled.")
                return
            
            # Extract code if user pasted full URL
            code = code.strip()
            if code.startswith("http"):
                from urllib.parse import urlparse, parse_qs
                try:
                    parsed = urlparse(code)
                    params = parse_qs(parsed.query)
                    if "code" in params:
                        code = params["code"][0]
                        log.log_info("Extracted authorization code from URL")
                except Exception as e:
                    log.log_warn(f"Failed to parse URL, using input as-is: {e}")
            
            # Exchange code for tokens
            log.log_info("Exchanging authorization code for tokens...")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                tokens = loop.run_until_complete(
                    exchange_code_for_tokens(code, verifier)
                )
            finally:
                loop.close()
            
            if tokens.get("error"):
                error_msg = tokens.get('error_description', tokens['error'])
                log.log_error(f"OAuth token exchange failed: {error_msg}")
                QMessageBox.critical(self, "Authentication Failed", 
                    f"Token exchange failed: {error_msg}")
                return
            
            # Extract account ID
            account_id = extract_account_id(tokens)
            
            # Create JSON credentials and store
            credentials_json = create_credentials_json(tokens, account_id)
            self.key_edit.setText(credentials_json)
            
            log.log_info("OAuth authentication successful - token stored in API key field")
            QMessageBox.information(self, "Authentication Successful", 
                "Successfully authenticated with ChatGPT Pro/Plus!\n\n"
                "The OAuth token has been stored. Click OK to save the provider.")
            
        except ImportError as e:
            log.log_error(f"OAuth import error: {e}")
            QMessageBox.critical(self, "Missing Dependencies", 
                f"OAuth authentication requires additional packages.\n\n"
                f"Please install: pip install aiohttp\n\n"
                f"Error: {e}")
        except Exception as e:
            log.log_error(f"OAuth authentication error: {e}")
            QMessageBox.critical(self, "Authentication Error", 
                f"An error occurred during authentication:\n\n{e}")


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
        self.transport_combo.addItem("Streamable HTTP", "streamablehttp")
        layout.addWidget(self.transport_combo)

        # Update URL placeholder based on transport
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

        # Update URL placeholder based on transport
        if transport_data == "sse":
            self.url_edit.setPlaceholderText("e.g., http://localhost:8000/sse")
        else:  # streamablehttp
            self.url_edit.setPlaceholderText("e.g., http://localhost:8000/mcp")
    
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
                    provider.get('provider_type', 'openai_platform'),
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

                # Validate required fields (URL not required for CLI-based providers)
                is_cli_provider = data.get('provider_type') == 'anthropic_cli'
                if not data['name'] or not data['model']:
                    self.show_error("Validation Error", "Name and Model are required fields.")
                    return
                if not is_cli_provider and not data['url']:
                    self.show_error("Validation Error", "URL is required for this provider type.")
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

                # Validate required fields (URL not required for CLI-based providers)
                is_cli_provider = data.get('provider_type') == 'anthropic_cli'
                if not data['name'] or not data['model']:
                    self.show_error("Validation Error", "Name and Model are required fields.")
                    return
                if not is_cli_provider and not data['url']:
                    self.show_error("Validation Error", "URL is required for this provider type.")
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
            # Some providers don't require API keys (Ollama runs locally, Claude Code uses CLI auth)
            provider_type = provider.get('provider_type', '')
            requires_key = provider_type not in ('ollama', 'anthropic_cli', 'anthropic_oauth', 'openai_oauth')
            if not provider.get('api_key') and requires_key:
                self.view.set_llm_test_status('failure',
                    f"Provider '{provider['name']}' has no API key configured. "
                    "Please edit the provider and add an API key before testing.")
                return

            # Note: OAuth providers (anthropic_oauth, openai_oauth) authenticate via the
            # Provider Edit dialog's Authenticate button. No special handling needed here.

            # Disable the test button and show testing state
            self.view.set_llm_test_enabled(False)
            self.view.set_llm_test_status('testing', f"Testing {provider['name']}...")

            # Create and start the test worker
            self.test_worker = ProviderTestWorker(provider)
            self.test_worker.test_completed.connect(self.on_provider_test_completed)
            self.test_worker.finished.connect(self.test_worker.deleteLater)
            self.test_worker.start()

        except Exception as e:
            self.view.set_llm_test_enabled(True)
            self.view.set_llm_test_status('failure', f"Test failed: {str(e)}")

    def on_provider_test_completed(self, success, message):
        """Handle provider test completion"""
        # Re-enable the test button
        self.view.set_llm_test_enabled(True)

        # Update status indicator
        if success:
            self.view.set_llm_test_status('success', message)
        else:
            self.view.set_llm_test_status('failure', message)
    
    
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
                self.view.set_mcp_test_status('failure',
                    f"Provider '{provider['name']}' has no URL configured. "
                    "Please edit the provider and add a URL before testing.")
                return

            # Disable the test button and show testing state
            self.view.set_mcp_test_enabled(False)
            self.view.set_mcp_test_status('testing', f"Testing {provider['name']}...")

            # Create and start the MCP test worker
            self.mcp_test_worker = MCPTestWorker(provider)
            self.mcp_test_worker.test_completed.connect(self.on_mcp_test_completed)
            self.mcp_test_worker.finished.connect(self.mcp_test_worker.deleteLater)
            self.mcp_test_worker.start()

        except Exception as e:
            self.view.set_mcp_test_enabled(True)
            self.view.set_mcp_test_status('failure', f"Test failed: {str(e)}")

    def on_mcp_test_completed(self, success, message, test_data):
        """Handle MCP test completion"""
        # Re-enable the test button
        self.view.set_mcp_test_enabled(True)

        # Update status indicator
        if success:
            self.view.set_mcp_test_status('success', message)
        else:
            self.view.set_mcp_test_status('failure', message)
    
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