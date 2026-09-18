//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//  \ V  V /  __/ (_| |\ V /| | (_| | ||  __/
//   \_/\_/ \___|\__,_| \_/ |_|\__,_|\__\___|
//
//  Copyright © 2016 - 2026 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

package clients

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConfigFromEnv(t *testing.T) {
	envVars := []string{
		EnvTenancyOCID, EnvUserOCID, EnvFingerprint, EnvPrivateKey, EnvPrivateKeyFile,
		EnvPrivateKeyPassphrase, EnvSecurityToken, EnvSecurityTokenFile, EnvRegion, EnvCompartmentID,
	}
	clearEnv := func(t *testing.T) {
		for _, name := range envVars {
			t.Setenv(name, "")
		}
	}

	t.Run("empty environment", func(t *testing.T) {
		clearEnv(t)
		cfg, err := ConfigFromEnv()
		require.NoError(t, err)
		assert.Equal(t, Config{}, cfg)
	})

	t.Run("inline private key", func(t *testing.T) {
		clearEnv(t)
		t.Setenv(EnvTenancyOCID, testTenancy)
		t.Setenv(EnvUserOCID, testUser)
		t.Setenv(EnvFingerprint, testFingerprint)
		t.Setenv(EnvPrivateKey, "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----\n")
		t.Setenv(EnvPrivateKeyPassphrase, "secret")
		t.Setenv(EnvRegion, testRegion)
		t.Setenv(EnvCompartmentID, testCompartment)

		cfg, err := ConfigFromEnv()
		require.NoError(t, err)
		assert.Equal(t, Config{
			TenancyOCID:          testTenancy,
			UserOCID:             testUser,
			Fingerprint:          testFingerprint,
			PrivateKeyPEM:        "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----\n",
			PrivateKeyPassphrase: "secret",
			Region:               testRegion,
			CompartmentID:        testCompartment,
		}, cfg)
	})

	t.Run("private key file", func(t *testing.T) {
		clearEnv(t)
		keyFile := filepath.Join(t.TempDir(), "key.pem")
		require.NoError(t, os.WriteFile(keyFile, []byte("pem-from-file"), 0o600))
		t.Setenv(EnvPrivateKeyFile, keyFile)

		cfg, err := ConfigFromEnv()
		require.NoError(t, err)
		assert.Equal(t, "pem-from-file", cfg.PrivateKeyPEM)
	})

	t.Run("inline private key wins over file", func(t *testing.T) {
		clearEnv(t)
		t.Setenv(EnvPrivateKey, "inline")
		t.Setenv(EnvPrivateKeyFile, filepath.Join(t.TempDir(), "does-not-exist.pem"))

		cfg, err := ConfigFromEnv()
		require.NoError(t, err)
		assert.Equal(t, "inline", cfg.PrivateKeyPEM)
	})

	t.Run("unreadable private key file", func(t *testing.T) {
		clearEnv(t)
		t.Setenv(EnvPrivateKeyFile, filepath.Join(t.TempDir(), "does-not-exist.pem"))

		_, err := ConfigFromEnv()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "read OCI_PRIVATE_KEY_FILE")
	})

	t.Run("session token settings", func(t *testing.T) {
		clearEnv(t)
		t.Setenv(EnvSecurityToken, "token")
		t.Setenv(EnvSecurityTokenFile, "/path/to/token")

		cfg, err := ConfigFromEnv()
		require.NoError(t, err)
		assert.Equal(t, "token", cfg.SecurityToken)
		assert.Equal(t, "/path/to/token", cfg.SecurityTokenFile)
	})
}

func TestDecodePrivateKeyHeader(t *testing.T) {
	_, keyPEM := newTestKey(t)

	t.Run("base64 encoded PEM", func(t *testing.T) {
		decoded, err := decodePrivateKeyHeader(base64.StdEncoding.EncodeToString([]byte(keyPEM)))
		require.NoError(t, err)
		assert.Equal(t, keyPEM, string(decoded))
	})

	t.Run("PEM with escaped newlines", func(t *testing.T) {
		escaped := strings.ReplaceAll(keyPEM, "\n", `\n`)
		decoded, err := decodePrivateKeyHeader(escaped)
		require.NoError(t, err)
		assert.Equal(t, keyPEM, string(decoded))
	})

	t.Run("plain PEM", func(t *testing.T) {
		decoded, err := decodePrivateKeyHeader(keyPEM)
		require.NoError(t, err)
		assert.Equal(t, strings.TrimSpace(keyPEM), string(decoded))
	})

	t.Run("garbage", func(t *testing.T) {
		_, err := decodePrivateKeyHeader("not base64 and not pem!")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "X-Oci-Private-Key must contain a PEM private key")
	})
}

func TestParsePrivateKey(t *testing.T) {
	key, keyPEM := newTestKey(t)

	parsed, err := parsePrivateKey([]byte(keyPEM), "")
	require.NoError(t, err)
	assert.True(t, key.Equal(parsed))

	_, err = parsePrivateKey([]byte("garbage"), "")
	require.Error(t, err)
}

func TestKeyProvider(t *testing.T) {
	key, _ := newTestKey(t)
	p := keyProvider{keyID: "ST$token", privateKey: key}

	keyID, err := p.KeyID()
	require.NoError(t, err)
	assert.Equal(t, "ST$token", keyID)

	privateKey, err := p.PrivateRSAKey()
	require.NoError(t, err)
	assert.Same(t, key, privateKey)
}
