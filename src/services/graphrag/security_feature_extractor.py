#!/usr/bin/env python3

import re
from typing import Iterable, Optional

from .security_features import SecurityFeatures

try:
    import binaryninja
    log = binaryninja.log.Logger(0, "BinAssist")
except ImportError:
    class MockLog:
        @staticmethod
        def log_info(msg): print(f"[BinAssist] INFO: {msg}")
        @staticmethod
        def log_warn(msg): print(f"[BinAssist] WARN: {msg}")
        @staticmethod
        def log_error(msg): print(f"[BinAssist] ERROR: {msg}")
    log = MockLog()


class SecurityFeatureExtractor:
    """Extracts security-relevant features from Binary Ninja functions."""

    NETWORK_APIS = {
        "socket", "bind", "listen", "accept", "connect", "close", "shutdown",
        "setsockopt", "getsockopt", "getpeername", "getsockname",
        "send", "sendto", "sendmsg", "recv", "recvfrom", "recvmsg",
        "read", "write",
        "select", "poll", "epoll_create", "epoll_ctl", "epoll_wait",
        "kqueue", "kevent",
        "WSAStartup", "WSACleanup", "WSAGetLastError", "WSASetLastError",
        "closesocket",
        "WSASocket", "WSASocketA", "WSASocketW",
        "WSAConnect", "WSAConnectByName", "WSAConnectByNameA", "WSAConnectByNameW",
        "WSAConnectByList",
        "WSAAccept",
        "WSASend", "WSASendTo", "WSASendMsg", "WSASendDisconnect",
        "WSARecv", "WSARecvFrom", "WSARecvMsg", "WSARecvDisconnect",
        "WSAAsyncSelect", "WSAEventSelect", "WSACreateEvent", "WSACloseEvent",
        "WSAWaitForMultipleEvents", "WSAEnumNetworkEvents", "WSAResetEvent", "WSASetEvent",
        "WSAIoctl", "WSAGetOverlappedResult",
        "WSAAddressToString", "WSAAddressToStringA", "WSAAddressToStringW",
        "WSAStringToAddress", "WSAStringToAddressA", "WSAStringToAddressW",
        "WSAEnumProtocols", "WSAEnumProtocolsA", "WSAEnumProtocolsW",
        "getaddrinfo", "GetAddrInfo", "GetAddrInfoA", "GetAddrInfoW", "GetAddrInfoEx",
        "freeaddrinfo", "FreeAddrInfo", "FreeAddrInfoW",
        "gethostbyname", "gethostbyaddr", "gethostname",
        "getnameinfo", "GetNameInfo", "GetNameInfoW",
        "inet_addr", "inet_ntoa", "inet_pton", "inet_ntop",
        "InetPton", "InetPtonA", "InetPtonW", "InetNtop", "InetNtopA", "InetNtopW",
        "htons", "htonl", "ntohs", "ntohl",
        "SSL_connect", "SSL_accept", "SSL_read", "SSL_write", "SSL_new", "SSL_free",
        "SSL_CTX_new", "SSL_CTX_free", "SSL_set_fd", "SSL_shutdown",
        "SSL_library_init", "SSL_load_error_strings",
        "TLS_client_method", "TLS_server_method",
        "AcquireCredentialsHandle", "AcquireCredentialsHandleA", "AcquireCredentialsHandleW",
        "InitializeSecurityContext", "InitializeSecurityContextA", "InitializeSecurityContextW",
        "AcceptSecurityContext",
        "EncryptMessage", "DecryptMessage",
        "WinHttpOpen", "WinHttpConnect", "WinHttpOpenRequest",
        "WinHttpSendRequest", "WinHttpReceiveResponse",
        "WinHttpReadData", "WinHttpWriteData", "WinHttpQueryHeaders",
        "WinHttpCloseHandle", "WinHttpSetOption", "WinHttpQueryOption",
        "WinHttpCrackUrl", "WinHttpCreateUrl",
        "InternetOpen", "InternetOpenA", "InternetOpenW",
        "InternetConnect", "InternetConnectA", "InternetConnectW",
        "InternetOpenUrl", "InternetOpenUrlA", "InternetOpenUrlW",
        "InternetReadFile", "InternetReadFileEx",
        "InternetWriteFile",
        "InternetCloseHandle", "InternetSetOption", "InternetQueryOption",
        "HttpOpenRequest", "HttpOpenRequestA", "HttpOpenRequestW",
        "HttpSendRequest", "HttpSendRequestA", "HttpSendRequestW",
        "HttpSendRequestEx", "HttpSendRequestExA", "HttpSendRequestExW",
        "HttpQueryInfo", "HttpQueryInfoA", "HttpQueryInfoW",
        "HttpAddRequestHeaders", "HttpEndRequest",
        "FtpOpenFile", "FtpGetFile", "FtpPutFile", "FtpDeleteFile",
        "FtpCreateDirectory", "FtpRemoveDirectory", "FtpFindFirstFile",
        "curl_easy_init", "curl_easy_perform", "curl_easy_cleanup",
        "curl_easy_setopt", "curl_easy_getinfo",
        "curl_multi_init", "curl_multi_add_handle", "curl_multi_perform"
    }

    FILE_IO_APIS = {
        "open", "close", "read", "write", "lseek", "pread", "pwrite",
        "creat", "dup", "dup2", "fcntl", "ioctl",
        "stat", "fstat", "lstat", "fstatat",
        "access", "faccessat", "chmod", "fchmod", "chown", "fchown",
        "truncate", "ftruncate",
        "mmap", "munmap", "msync",
        "fopen", "fclose", "fread", "fwrite", "fgets", "fputs", "fgetc", "fputc",
        "fprintf", "fscanf", "fseek", "ftell", "fflush", "rewind", "feof", "ferror",
        "freopen", "fdopen", "fileno", "setvbuf", "setbuf",
        "getc", "putc", "ungetc", "getchar", "putchar",
        "opendir", "closedir", "readdir", "readdir_r", "scandir", "seekdir", "telldir",
        "mkdir", "mkdirat", "rmdir", "chdir", "fchdir", "getcwd",
        "realpath", "basename", "dirname",
        "rename", "renameat", "remove", "unlink", "unlinkat",
        "link", "linkat", "symlink", "symlinkat", "readlink", "readlinkat",
        "CreateFile", "CreateFileA", "CreateFileW",
        "CreateFile2",
        "OpenFile",
        "ReadFile", "ReadFileEx", "ReadFileScatter",
        "WriteFile", "WriteFileEx", "WriteFileGather",
        "FlushFileBuffers",
        "SetFilePointer", "SetFilePointerEx",
        "SetEndOfFile",
        "GetFileSize", "GetFileSizeEx",
        "GetFileType",
        "GetFileTime", "SetFileTime",
        "GetFileInformationByHandle", "GetFileInformationByHandleEx",
        "SetFileInformationByHandle",
        "GetFileAttributes", "GetFileAttributesA", "GetFileAttributesW",
        "GetFileAttributesEx", "GetFileAttributesExA", "GetFileAttributesExW",
        "SetFileAttributes", "SetFileAttributesA", "SetFileAttributesW",
        "LockFile", "LockFileEx", "UnlockFile", "UnlockFileEx",
        "DeleteFile", "DeleteFileA", "DeleteFileW",
        "CopyFile", "CopyFileA", "CopyFileW",
        "CopyFileEx", "CopyFileExA", "CopyFileExW",
        "MoveFile", "MoveFileA", "MoveFileW",
        "MoveFileEx", "MoveFileExA", "MoveFileExW",
        "MoveFileWithProgress", "MoveFileWithProgressA", "MoveFileWithProgressW",
        "ReplaceFile", "ReplaceFileA", "ReplaceFileW",
        "FindFirstFile", "FindFirstFileA", "FindFirstFileW",
        "FindFirstFileEx", "FindFirstFileExA", "FindFirstFileExW",
        "FindNextFile", "FindNextFileA", "FindNextFileW",
        "FindClose",
        "SearchPath", "SearchPathA", "SearchPathW",
        "CreateDirectory", "CreateDirectoryA", "CreateDirectoryW",
        "CreateDirectoryEx", "CreateDirectoryExA", "CreateDirectoryExW",
        "RemoveDirectory", "RemoveDirectoryA", "RemoveDirectoryW",
        "SetCurrentDirectory", "SetCurrentDirectoryA", "SetCurrentDirectoryW",
        "GetCurrentDirectory", "GetCurrentDirectoryA", "GetCurrentDirectoryW",
        "GetFullPathName", "GetFullPathNameA", "GetFullPathNameW",
        "GetLongPathName", "GetLongPathNameA", "GetLongPathNameW",
        "GetShortPathName", "GetShortPathNameA", "GetShortPathNameW",
        "GetTempPath", "GetTempPathA", "GetTempPathW",
        "GetTempFileName", "GetTempFileNameA", "GetTempFileNameW",
        "PathFileExists", "PathFileExistsA", "PathFileExistsW",
        "CloseHandle", "DuplicateHandle",
        "CreateFileMapping", "CreateFileMappingA", "CreateFileMappingW",
        "OpenFileMapping", "OpenFileMappingA", "OpenFileMappingW",
        "MapViewOfFile", "MapViewOfFileEx", "UnmapViewOfFile",
        "GetOverlappedResult", "GetOverlappedResultEx",
        "CancelIo", "CancelIoEx", "CancelSynchronousIo",
        "CreateFileTransacted", "CreateFileTransactedA", "CreateFileTransactedW",
        "DeleteFileTransacted", "DeleteFileTransactedA", "DeleteFileTransactedW"
    }

    CRYPTO_APIS = {
        "CryptAcquireContext", "CryptReleaseContext",
        "CryptGenKey", "CryptDeriveKey", "CryptDestroyKey",
        "CryptEncrypt", "CryptDecrypt",
        "CryptCreateHash", "CryptHashData", "CryptDestroyHash",
        "CryptSignHash", "CryptVerifySignature",
        "CryptImportKey", "CryptExportKey",
        "BCryptOpenAlgorithmProvider", "BCryptCloseAlgorithmProvider",
        "BCryptGenerateKeyPair", "BCryptEncrypt", "BCryptDecrypt",
        "BCryptCreateHash", "BCryptHashData", "BCryptFinishHash",
        "EVP_EncryptInit", "EVP_EncryptUpdate", "EVP_EncryptFinal",
        "EVP_DecryptInit", "EVP_DecryptUpdate", "EVP_DecryptFinal",
        "EVP_DigestInit", "EVP_DigestUpdate", "EVP_DigestFinal",
        "AES_encrypt", "AES_decrypt", "AES_set_encrypt_key", "AES_set_decrypt_key",
        "RSA_public_encrypt", "RSA_private_decrypt",
        "MD5_Init", "MD5_Update", "MD5_Final",
        "SHA1_Init", "SHA1_Update", "SHA1_Final",
        "SHA256_Init", "SHA256_Update", "SHA256_Final"
    }

    PROCESS_APIS = {
        "fork", "exec", "execl", "execle", "execlp", "execv", "execve", "execvp",
        "system", "popen", "pclose",
        "kill", "waitpid", "wait",
        "CreateProcess", "CreateProcessA", "CreateProcessW",
        "CreateProcessAsUser", "CreateProcessWithLogon",
        "ShellExecute", "ShellExecuteA", "ShellExecuteW",
        "ShellExecuteEx", "ShellExecuteExA", "ShellExecuteExW",
        "WinExec", "LoadLibrary", "LoadLibraryA", "LoadLibraryW",
        "GetProcAddress", "FreeLibrary",
        "OpenProcess", "TerminateProcess",
        "VirtualAlloc", "VirtualAllocEx", "VirtualProtect",
        "WriteProcessMemory", "ReadProcessMemory",
        "CreateRemoteThread", "CreateRemoteThreadEx"
    }

    DANGEROUS_FUNCTIONS = {
        "strcpy": "BUFFER_OVERFLOW_RISK",
        "strcat": "BUFFER_OVERFLOW_RISK",
        "sprintf": "BUFFER_OVERFLOW_RISK",
        "vsprintf": "BUFFER_OVERFLOW_RISK",
        "gets": "BUFFER_OVERFLOW_RISK",
        "scanf": "BUFFER_OVERFLOW_RISK",
        "fscanf": "BUFFER_OVERFLOW_RISK",
        "sscanf": "BUFFER_OVERFLOW_RISK",
        "wcscpy": "BUFFER_OVERFLOW_RISK",
        "wcscat": "BUFFER_OVERFLOW_RISK",
        "lstrcpy": "BUFFER_OVERFLOW_RISK",
        "lstrcpyA": "BUFFER_OVERFLOW_RISK",
        "lstrcpyW": "BUFFER_OVERFLOW_RISK",
        "lstrcat": "BUFFER_OVERFLOW_RISK",
        "lstrcatA": "BUFFER_OVERFLOW_RISK",
        "lstrcatW": "BUFFER_OVERFLOW_RISK",
        "StrCpy": "BUFFER_OVERFLOW_RISK",
        "StrCat": "BUFFER_OVERFLOW_RISK",
        "_tcscpy": "BUFFER_OVERFLOW_RISK",
        "_tcscat": "BUFFER_OVERFLOW_RISK",
        "_mbscpy": "BUFFER_OVERFLOW_RISK",
        "_mbscat": "BUFFER_OVERFLOW_RISK",
        "printf": "FORMAT_STRING_RISK",
        "fprintf": "FORMAT_STRING_RISK",
        "wprintf": "FORMAT_STRING_RISK",
        "syslog": "FORMAT_STRING_RISK",
        "system": "COMMAND_INJECTION_RISK",
        "popen": "COMMAND_INJECTION_RISK",
        "_popen": "COMMAND_INJECTION_RISK",
        "wpopen": "COMMAND_INJECTION_RISK",
        "execl": "COMMAND_INJECTION_RISK",
        "execle": "COMMAND_INJECTION_RISK",
        "execlp": "COMMAND_INJECTION_RISK",
        "execv": "COMMAND_INJECTION_RISK",
        "execve": "COMMAND_INJECTION_RISK",
        "execvp": "COMMAND_INJECTION_RISK",
        "WinExec": "COMMAND_INJECTION_RISK",
        "ShellExecute": "COMMAND_INJECTION_RISK",
        "ShellExecuteA": "COMMAND_INJECTION_RISK",
        "ShellExecuteW": "COMMAND_INJECTION_RISK",
        "ShellExecuteEx": "COMMAND_INJECTION_RISK",
        "ShellExecuteExA": "COMMAND_INJECTION_RISK",
        "ShellExecuteExW": "COMMAND_INJECTION_RISK",
        "atoi": "INTEGER_OVERFLOW_RISK",
        "atol": "INTEGER_OVERFLOW_RISK",
        "atoll": "INTEGER_OVERFLOW_RISK",
        "strtol": "INTEGER_OVERFLOW_RISK",
        "strtoul": "INTEGER_OVERFLOW_RISK",
        "access": "RACE_CONDITION_RISK",
        "stat": "RACE_CONDITION_RISK",
        "alloca": "MEMORY_CORRUPTION_RISK",
        "_alloca": "MEMORY_CORRUPTION_RISK",
        "rand": "WEAK_RANDOM_RISK",
        "srand": "WEAK_RANDOM_RISK",
        "random": "WEAK_RANDOM_RISK",
        "MD5_Init": "WEAK_CRYPTO_RISK",
        "MD5_Update": "WEAK_CRYPTO_RISK",
        "MD5_Final": "WEAK_CRYPTO_RISK",
        "MD5": "WEAK_CRYPTO_RISK",
        "SHA1_Init": "WEAK_CRYPTO_RISK",
        "SHA1_Update": "WEAK_CRYPTO_RISK",
        "SHA1_Final": "WEAK_CRYPTO_RISK",
        "DES_encrypt": "WEAK_CRYPTO_RISK",
        "DES_decrypt": "WEAK_CRYPTO_RISK"
    }

    IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
    URL_PATTERN = re.compile(r"^https?://[\w.-]+(?:/[\w./?%&=-]*)?$", re.IGNORECASE)
    UNIX_PATH_PATTERN = re.compile(r"^/[a-zA-Z0-9/_.-]+$")
    WINDOWS_PATH_PATTERN = re.compile(r"^[A-Za-z]:\\\\[^<>:\"|?*]+$")
    DOMAIN_PATTERN = re.compile(
        r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?"
        r"(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$"
    )
    REGISTRY_PATTERN = re.compile(r"^(HKEY_|HK[A-Z]{2})[\\\\A-Za-z0-9_-]+$")

    def __init__(self, binary_view):
        self.binary_view = binary_view

    def extract_features(self, function) -> SecurityFeatures:
        features = SecurityFeatures()
        if function is None or self.binary_view is None:
            return features

        try:
            self._extract_api_calls(function, features)
            self._extract_string_references(function, features)
            features.calculate_activity_profile()
            features.calculate_risk_level()
        except Exception as e:
            log.log_warn(f"Error extracting security features: {e}")

        return features

    def _extract_api_calls(self, function, features: SecurityFeatures) -> None:
        try:
            callees = getattr(function, "callees", None) or []
            for callee in callees:
                name = getattr(callee, "name", None)
                if not name:
                    continue
                normalized = self._normalize_function_name(name)

                if name in self.NETWORK_APIS or normalized in self.NETWORK_APIS:
                    features.add_network_api(normalized)
                if name in self.FILE_IO_APIS or normalized in self.FILE_IO_APIS:
                    features.add_file_io_api(normalized)
                if name in self.CRYPTO_APIS or normalized in self.CRYPTO_APIS:
                    features.add_crypto_api(normalized)
                if name in self.PROCESS_APIS or normalized in self.PROCESS_APIS:
                    features.add_process_api(normalized)

                vuln_type = self.DANGEROUS_FUNCTIONS.get(name) or self.DANGEROUS_FUNCTIONS.get(normalized)
                if vuln_type:
                    features.add_dangerous_function(normalized, vuln_type)
        except Exception as e:
            log.log_warn(f"Error extracting API calls: {e}")

    def _normalize_function_name(self, name: str) -> str:
        if not name:
            return name

        normalized = name
        if normalized.startswith("__imp_"):
            normalized = normalized[6:]

        while normalized.startswith("_") and len(normalized) > 1:
            normalized = normalized[1:]

        at_index = normalized.rfind("@")
        if at_index > 0:
            suffix = normalized[at_index + 1 :]
            if suffix.isdigit():
                normalized = normalized[:at_index]

        return normalized

    def _extract_string_references(self, function, features: SecurityFeatures) -> None:
        try:
            strings = getattr(self.binary_view, "strings", None)
            if not strings:
                return

            for string_ref in strings:
                value = getattr(string_ref, "value", None)
                if not value or len(value) < 3:
                    continue

                start = getattr(string_ref, "start", None)
                if start is None:
                    continue

                if not self._string_referenced_in_function(function, start):
                    continue

                self._classify_string(value, features)
        except Exception as e:
            log.log_warn(f"Error extracting string references: {e}")

    def _string_referenced_in_function(self, function, address: int) -> bool:
        try:
            refs = self.binary_view.get_code_refs(address)
            if not refs:
                return False
            for ref in refs:
                ref_addr = getattr(ref, "address", None)
                if ref_addr is None:
                    ref_addr = getattr(ref, "source", None)
                if ref_addr is None:
                    continue
                if self._address_in_function(function, ref_addr):
                    return True
            return False
        except Exception:
            return False

    def _address_in_function(self, function, address: int) -> bool:
        try:
            for block in function.basic_blocks:
                if block.start <= address < block.end:
                    return True
        except Exception:
            return False
        return False

    def _classify_string(self, value: str, features: SecurityFeatures) -> None:
        if not value:
            return

        value = value.strip()
        if len(value) < 3:
            return

        if self.IP_PATTERN.match(value):
            if self._is_valid_ip(value):
                features.add_ip_address(value)
        elif self.URL_PATTERN.match(value):
            features.add_url(value)
        elif self.REGISTRY_PATTERN.match(value):
            features.add_registry_key(value)
        elif self.UNIX_PATH_PATTERN.match(value):
            features.add_file_path(value)
        elif self.WINDOWS_PATH_PATTERN.match(value):
            features.add_file_path(value)
        elif self.DOMAIN_PATTERN.match(value):
            if not self._is_common_non_domain(value):
                features.add_domain(value)

    def _is_valid_ip(self, ip: str) -> bool:
        try:
            parts = ip.split(".")
            if len(parts) != 4:
                return False
            for part in parts:
                octet = int(part)
                if octet < 0 or octet > 255:
                    return False
            return True
        except Exception:
            return False

    def _is_common_non_domain(self, value: str) -> bool:
        lower = value.lower()
        return (
            lower.endswith(".dll")
            or lower.endswith(".exe")
            or lower.endswith(".sys")
            or lower.endswith(".lib")
            or lower.endswith(".obj")
            or lower.endswith(".pdb")
            or lower.endswith(".h")
            or lower.endswith(".c")
            or lower.endswith(".cpp")
            or lower == "version.rc"
            or ("microsoft.com" in lower and "schema" in lower)
        )

    # Static Analysis Helpers
    @staticmethod
    def suggests_network_activity(function_name: Optional[str]) -> bool:
        if not function_name:
            return False
        lower = function_name.lower()
        return any(
            key in lower
            for key in ("socket", "connect", "send", "recv", "http", "download", "upload", "network", "client", "server")
        )

    @staticmethod
    def suggests_file_activity(function_name: Optional[str]) -> bool:
        if not function_name:
            return False
        lower = function_name.lower()
        return any(
            key in lower
            for key in ("file", "read", "write", "open", "save", "load", "config", "log")
        )

    @staticmethod
    def suggests_crypto_activity(function_name: Optional[str]) -> bool:
        if not function_name:
            return False
        lower = function_name.lower()
        return any(
            key in lower
            for key in ("crypt", "encrypt", "decrypt", "hash", "aes", "rsa", "sha", "md5")
        )
