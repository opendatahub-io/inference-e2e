MANIFEST_REPO ?= https://github.com/opendatahub-io/llm-d-conformance-manifests.git
BRANCH        ?=
MANIFEST_REF  ?= $(if $(BRANCH),$(BRANCH),main)
MANIFEST_DIR  ?= deploy/manifests
NAMESPACE     ?= llm-conformance-test
TESTCASE_DIR  ?= configs/testcases
REPORT_DIR    ?= reports
PLATFORM      ?= any
MODEL_SOURCE  ?= hf
KUBECONFIG    ?= $(HOME)/.kube/config

# Test flags
TESTCASE      ?=
PROFILE       ?=
MODEL         ?=
MOCK          ?=
PULL_SECRET   ?=
BEARER_TOKEN  ?=
STORAGE_CLASS ?=
STORAGE_SIZE  ?=
EXTRA_FLAGS   ?=

# Build pytest args
PYTEST_ARGS := tests/test_conformance.py \
	--platform $(PLATFORM) \
	--namespace $(NAMESPACE) \
	--kubeconfig $(KUBECONFIG) \
	--testcase-dir $(TESTCASE_DIR) \
	--report-dir $(REPORT_DIR) \
	--model-source $(MODEL_SOURCE)

ifdef TESTCASE
  PYTEST_ARGS += --testcase $(TESTCASE)
endif
ifdef PROFILE
  PYTEST_ARGS += --profile $(PROFILE)
endif
ifdef MODEL
  PYTEST_ARGS += --model $(MODEL)
endif
ifdef MOCK
  PYTEST_ARGS += --mock $(MOCK)
endif
ifdef PULL_SECRET
  PYTEST_ARGS += --pull-secret $(PULL_SECRET)
endif
ifdef BEARER_TOKEN
  PYTEST_ARGS += --bearer-token $(BEARER_TOKEN)
endif
ifdef STORAGE_CLASS
  PYTEST_ARGS += --storage-class $(STORAGE_CLASS)
endif
ifdef STORAGE_SIZE
  PYTEST_ARGS += --storage-size $(STORAGE_SIZE)
endif

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Clone manifest repo (MANIFEST_REF=branch, or interactive if not set)
	@if [ -t 0 ] && [ "$(origin MANIFEST_REF)" = "file" ] && [ -z "$(BRANCH)" ]; then \
		echo "Fetching branches from $(MANIFEST_REPO)..."; \
		BRANCHES=$$(git ls-remote --heads $(MANIFEST_REPO) 2>/dev/null | sed 's|.*refs/heads/||' | sort); \
		echo ""; \
		echo "Available branches:"; \
		i=1; for b in $$BRANCHES; do echo "  $$i) $$b"; i=$$((i+1)); done; \
		echo ""; \
		printf "Select branch number (or press Enter for main): "; \
		read choice; \
		if [ -n "$$choice" ]; then \
			SELECTED=$$(echo "$$BRANCHES" | sed -n "$${choice}p"); \
			if [ -z "$$SELECTED" ]; then echo "Invalid selection"; exit 1; fi; \
		else \
			SELECTED="main"; \
		fi; \
		$(MAKE) _do-setup MANIFEST_REF=$$SELECTED; \
	else \
		$(MAKE) _do-setup; \
	fi

.PHONY: _do-setup
_do-setup:
	@mkdir -p $(MANIFEST_DIR)
	@rm -rf $(MANIFEST_DIR)/*.yaml
	@set -e; \
	tmpdir=$$(mktemp -d); \
	trap 'rm -rf "$$tmpdir"' EXIT; \
	clone_dir="$$tmpdir/repo"; \
	git clone --depth 1 --branch "$(MANIFEST_REF)" "$(MANIFEST_REPO)" "$$clone_dir"; \
	COMMIT=$$(git -C "$$clone_dir" rev-parse HEAD); \
	cp "$$clone_dir"/*.yaml "$(MANIFEST_DIR)/"; \
	printf 'branch: %s\nrepo: %s\ncommit: %s\ndate: %s\n' \
		"$(MANIFEST_REF)" "$(MANIFEST_REPO)" "$$COMMIT" "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
		> "$(MANIFEST_DIR)/.manifest-ref"; \
	COMMIT_SHORT=$$(printf '%.8s' "$$COMMIT"); \
	echo "Manifests ready (branch: $(MANIFEST_REF), commit: $$COMMIT_SHORT)"; \
	echo ""; \
	echo "Test cases:"; \
	for tc in $(TESTCASE_DIR)/*.yaml; do \
		name=$$(grep '^name:' $$tc | head -1 | sed 's/name: *//'); \
		manifest=$$(grep 'manifestPath:' $$tc | head -1 | sed 's/.*manifestPath: *//'); \
		if [ -f "$(MANIFEST_DIR)/$$manifest" ]; then \
			printf "  \033[32m✓\033[0m %-28s → %s\n" "$$name" "$$manifest"; \
		else \
			printf "  \033[31m✗\033[0m %-28s → %s (missing)\n" "$$name" "$$manifest"; \
		fi; \
	done

# Dry-run with a default (not -n) namespace: documents without one land in $(NAMESPACE),
# as the deployer applies them, and cross-namespace documents (e.g. MaaS policies) keep
# their own. Manifests whose kinds the cluster lacks (e.g. MaaS) are reported as skipped.
.PHONY: validate-manifests
validate-manifests: ## Validate all manifests against the cluster CRD (dry-run)
	@if [ ! -d $(MANIFEST_DIR) ] || [ -z "$$(ls $(MANIFEST_DIR)/*.yaml 2>/dev/null)" ]; then \
		echo "No manifests found — run 'make setup' first"; exit 1; \
	fi
	@echo "Validating manifests against cluster (dry-run)..."
	@FAIL=0; \
	kubectl --kubeconfig $(KUBECONFIG) get namespace $(NAMESPACE) >/dev/null 2>&1 || \
		kubectl --kubeconfig $(KUBECONFIG) create namespace $(NAMESPACE) >/dev/null 2>&1; \
	KC=$$(mktemp); trap 'rm -f $$KC' EXIT; \
	kubectl --kubeconfig $(KUBECONFIG) config view --raw --minify --flatten > $$KC; \
	kubectl --kubeconfig $$KC config set-context --current --namespace=$(NAMESPACE) >/dev/null; \
	for f in $(MANIFEST_DIR)/*.yaml; do \
		name=$$(basename $$f); \
		out=$$(kubectl --kubeconfig $$KC apply --dry-run=server -f $$f 2>&1); \
		if [ $$? -eq 0 ]; then \
			printf "  \033[32m✓\033[0m %s\n" "$$name"; \
		elif echo "$$out" | grep -q "no matches for kind"; then \
			printf "  \033[33m⊘\033[0m %s (skipped: %s not installed on this cluster)\n" "$$name" \
				"$$(echo "$$out" | grep -o 'kind "[^"]*"' | head -1)"; \
		else \
			printf "  \033[31m✗\033[0m %s\n" "$$name"; \
			echo "$$out" | sed 's/^/    /'; \
			FAIL=1; \
		fi; \
	done; \
	if [ $$FAIL -eq 1 ]; then echo ""; echo "Validation failed"; exit 1; \
	else echo ""; echo "All manifests valid"; fi

.PHONY: sync
sync: ## Install dependencies with uv
	uv sync

.PHONY: test
test: ## Run conformance tests (TESTCASE=single-gpu)
	uv run pytest $(PYTEST_ARGS) $(EXTRA_FLAGS)

.PHONY: test-profile-smoke
test-profile-smoke: ## Run smoke profile
	uv run pytest $(PYTEST_ARGS) --profile configs/profiles/smoke.yaml

.PHONY: test-profile-all
test-profile-all: ## Run all conformance tests
	uv run pytest $(PYTEST_ARGS) --profile configs/profiles/all.yaml

.PHONY: test-profile-cache-aware
test-profile-cache-aware: ## Run cache-aware tests
	uv run pytest $(PYTEST_ARGS) --profile configs/profiles/cache-aware.yaml

.PHONY: test-profile-pd
test-profile-pd: ## Run P/D disaggregation tests
	uv run pytest $(PYTEST_ARGS) --profile configs/profiles/pd.yaml

.PHONY: test-profile-moe
test-profile-moe: ## Run MoE/DeepSeek tests
	uv run pytest $(PYTEST_ARGS) --profile configs/profiles/deepseek.yaml

.PHONY: test-profile-flow-control
test-profile-flow-control: ## Run flow control tests
	uv run pytest $(PYTEST_ARGS) --profile configs/profiles/flow-control.yaml

.PHONY: unittest
unittest: ## Run unit tests (no cluster needed)
	uv run pytest tests/ -v --ignore=tests/test_conformance.py

.PHONY: lint
lint: ## Run ruff linter
	uv run ruff check src/ tests/

.PHONY: format
format: ## Format code with ruff
	uv run ruff format src/ tests/

.PHONY: testcases
testcases: ## List available test cases
	@if [ -f $(MANIFEST_DIR)/.manifest-ref ]; then \
		echo "Manifests:"; cat $(MANIFEST_DIR)/.manifest-ref | sed 's/^/  /'; echo; \
	else \
		echo "Manifests: not set up (run make setup)"; echo; \
	fi
	@echo "Test cases:"
	@for f in $(TESTCASE_DIR)/*.yaml; do \
		name=$$(grep '^name:' $$f | head -1 | sed 's/name: *//'); \
		desc=$$(grep '^description:' $$f | head -1 | sed 's/description: *"//;s/"$$//'); \
		printf "  %-28s %s\n" "$$name" "$$desc"; \
	done

.PHONY: profiles
profiles: ## List available profiles
	@if [ -f $(MANIFEST_DIR)/.manifest-ref ]; then \
		echo "Manifests:"; cat $(MANIFEST_DIR)/.manifest-ref | sed 's/^/  /'; echo; \
	else \
		echo "Manifests: not set up (run make setup)"; echo; \
	fi
	@echo "Profiles:"
	@for f in configs/profiles/*.yaml; do \
		name=$$(grep '^name:' $$f | head -1 | sed 's/name: *//'); \
		desc=$$(grep '^description:' $$f | head -1 | sed 's/description: *"//;s/"$$//'); \
		printf "  %-20s %s\n" "$$name" "$$desc"; \
	done

.PHONY: clean
clean: ## Remove reports and temp files
	rm -rf reports/*.json reports/*.html __pycache__ .pytest_cache
