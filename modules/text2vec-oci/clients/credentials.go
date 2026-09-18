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
	"context"
	"crypto/rsa"
	"encoding/base64"
	"fmt"
	"os"
	"strings"

	"github.com/oracle/oci-go-sdk/v65/common"
	"github.com/pkg/errors"

	"github.com/weaviate/weaviate/usecases/modulecomponents"
)

// Environment variables read by the module at startup.
const (
	EnvTenancyOCID          = "OCI_TENANCY_OCID"
	EnvUserOCID             = "OCI_USER_OCID"
	EnvFingerprint          = "OCI_FINGERPRINT"
	EnvPrivateKey           = "OCI_PRIVATE_KEY"
	EnvPrivateKeyFile       = "OCI_PRIVATE_KEY_FILE"
	EnvPrivateKeyPassphrase = "OCI_PRIVATE_KEY_PASSPHRASE"
	EnvSecurityToken        = "OCI_SECURITY_TOKEN"
	EnvSecurityTokenFile    = "OCI_SECURITY_TOKEN_FILE"
	EnvRegion               = "OCI_REGION"
	EnvCompartmentID        = "OCI_COMPARTMENT_ID"
)

// Request headers that override the environment configuration per request,
// following the X-Aws-* convention of the text2vec-aws module.
const (
	HeaderTenancyOCID = "X-Oci-Tenancy-Ocid"
	HeaderUserOCID    = "X-Oci-User-Ocid"
	HeaderFingerprint = "X-Oci-Fingerprint"
	// HeaderPrivateKey carries the PEM private key either base64-encoded or
	// with its line breaks escaped as the two characters `\n`, so that it fits
	// a single header line.
	HeaderPrivateKey    = "X-Oci-Private-Key"
	HeaderSecurityToken = "X-Oci-Security-Token"
	HeaderRegion        = "X-Oci-Region"
	HeaderCompartmentID = "X-Oci-Compartment-Id"
)

// securityTokenKeyIDPrefix marks a session token based keyId in the OCI HTTP
// signature, as produced by the OCI SDKs for `oci session authenticate`.
const securityTokenKeyIDPrefix = "ST$"

// Config is the environment-provided module configuration. Credentials come
// in two flavours:
//
//   - API key: TenancyOCID, UserOCID, Fingerprint and the PEM private key
//     (signature keyId "tenancy/user/fingerprint")
//   - Security (session) token: the PEM session private key and the token
//     (signature keyId "ST$<token>"), as created by `oci session authenticate`
type Config struct {
	TenancyOCID          string
	UserOCID             string
	Fingerprint          string
	PrivateKeyPEM        string
	PrivateKeyPassphrase string
	SecurityToken        string
	// SecurityTokenFile is re-read on every request so a refreshed session
	// (`oci session refresh`) is picked up without restarting Weaviate.
	SecurityTokenFile string
	Region            string
	CompartmentID     string
}

// ConfigFromEnv reads the module configuration from the environment. It only
// fails when a referenced private key file cannot be read; missing values are
// reported at request time so the module can start without credentials.
func ConfigFromEnv() (Config, error) {
	cfg := Config{
		TenancyOCID:          os.Getenv(EnvTenancyOCID),
		UserOCID:             os.Getenv(EnvUserOCID),
		Fingerprint:          os.Getenv(EnvFingerprint),
		PrivateKeyPEM:        os.Getenv(EnvPrivateKey),
		PrivateKeyPassphrase: os.Getenv(EnvPrivateKeyPassphrase),
		SecurityToken:        os.Getenv(EnvSecurityToken),
		SecurityTokenFile:    os.Getenv(EnvSecurityTokenFile),
		Region:               os.Getenv(EnvRegion),
		CompartmentID:        os.Getenv(EnvCompartmentID),
	}
	if cfg.PrivateKeyPEM == "" {
		if file := os.Getenv(EnvPrivateKeyFile); file != "" {
			pemBytes, err := os.ReadFile(file)
			if err != nil {
				return Config{}, errors.Wrapf(err, "read %s", EnvPrivateKeyFile)
			}
			cfg.PrivateKeyPEM = string(pemBytes)
		}
	}
	return cfg, nil
}

// keyProvider is the common.KeyProvider handed to the OCI request signer.
type keyProvider struct {
	keyID      string
	privateKey *rsa.PrivateKey
}

var _ common.KeyProvider = keyProvider{}

func (p keyProvider) PrivateRSAKey() (*rsa.PrivateKey, error) {
	return p.privateKey, nil
}

func (p keyProvider) KeyID() (string, error) {
	return p.keyID, nil
}

func parsePrivateKey(pemData []byte, passphrase string) (*rsa.PrivateKey, error) {
	var password []byte
	if passphrase != "" {
		password = []byte(passphrase)
	}
	return common.PrivateKeyFromBytesWithPassword(pemData, password)
}

// decodePrivateKeyHeader accepts a PEM key that is either base64-encoded or
// has its newlines escaped as `\n`.
func decodePrivateKeyHeader(value string) ([]byte, error) {
	value = strings.TrimSpace(value)
	if strings.Contains(value, "-----BEGIN") {
		return []byte(strings.ReplaceAll(value, `\n`, "\n")), nil
	}
	decoded, err := base64.StdEncoding.DecodeString(value)
	if err != nil {
		return nil, fmt.Errorf("%s must contain a PEM private key, either base64-encoded or with newlines escaped as \\n", HeaderPrivateKey)
	}
	return decoded, nil
}

func (c *Client) getPrivateKey(ctx context.Context) (*rsa.PrivateKey, error) {
	if header := modulecomponents.GetValueFromContext(ctx, HeaderPrivateKey); header != "" {
		pemData, err := decodePrivateKeyHeader(header)
		if err != nil {
			return nil, err
		}
		key, err := parsePrivateKey(pemData, "")
		if err != nil {
			return nil, errors.Wrapf(err, "parse private key from %s header", HeaderPrivateKey)
		}
		return key, nil
	}
	if c.privateKey != nil {
		return c.privateKey, nil
	}
	return nil, errors.Errorf("no private key found neither in request header: %s "+
		"nor in environment variables under %s or %s", HeaderPrivateKey, EnvPrivateKey, EnvPrivateKeyFile)
}

func (c *Client) getSecurityToken(ctx context.Context) (string, error) {
	if token := modulecomponents.GetValueFromContext(ctx, HeaderSecurityToken); token != "" {
		return token, nil
	}
	if c.config.SecurityToken != "" {
		return c.config.SecurityToken, nil
	}
	if c.config.SecurityTokenFile != "" {
		token, err := os.ReadFile(c.config.SecurityTokenFile)
		if err != nil {
			return "", errors.Wrapf(err, "read %s", EnvSecurityTokenFile)
		}
		return strings.TrimSpace(string(token)), nil
	}
	return "", nil
}

func (c *Client) getKeyProvider(ctx context.Context) (keyProvider, error) {
	privateKey, err := c.getPrivateKey(ctx)
	if err != nil {
		return keyProvider{}, err
	}

	token, err := c.getSecurityToken(ctx)
	if err != nil {
		return keyProvider{}, err
	}
	if token != "" {
		return keyProvider{keyID: securityTokenKeyIDPrefix + token, privateKey: privateKey}, nil
	}

	tenancy := c.headerOrConfig(ctx, HeaderTenancyOCID, c.config.TenancyOCID)
	user := c.headerOrConfig(ctx, HeaderUserOCID, c.config.UserOCID)
	fingerprint := c.headerOrConfig(ctx, HeaderFingerprint, c.config.Fingerprint)

	var missing []string
	if tenancy == "" {
		missing = append(missing, fmt.Sprintf("%s (%s)", EnvTenancyOCID, HeaderTenancyOCID))
	}
	if user == "" {
		missing = append(missing, fmt.Sprintf("%s (%s)", EnvUserOCID, HeaderUserOCID))
	}
	if fingerprint == "" {
		missing = append(missing, fmt.Sprintf("%s (%s)", EnvFingerprint, HeaderFingerprint))
	}
	if len(missing) > 0 {
		return keyProvider{}, errors.Errorf("incomplete OCI API key credentials, missing: %s; "+
			"alternatively provide a session token via %s or %s (%s)",
			strings.Join(missing, ", "), EnvSecurityToken, EnvSecurityTokenFile, HeaderSecurityToken)
	}

	return keyProvider{
		keyID:      fmt.Sprintf("%s/%s/%s", tenancy, user, fingerprint),
		privateKey: privateKey,
	}, nil
}

func (c *Client) headerOrConfig(ctx context.Context, header, configValue string) string {
	if value := modulecomponents.GetValueFromContext(ctx, header); value != "" {
		return value
	}
	return configValue
}
