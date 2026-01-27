/**
 * Authentication API
 * Handles communication with the backend auth endpoints
 */

// Use Vite environment variable if available, otherwise default to localhost
// In production/Docker, VITE_API_URL should be set in docker-compose.yml
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Login user with username and password
 * @param {string} username 
 * @param {string} password 
 * @returns {Promise<Object>} User data with role
 */
export async function login(username, password) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
            credentials: 'include', // Include cookies for session management
        });

        if (!response.ok) {
            if (response.status === 401) {
                throw new Error('Invalid credentials');
            }
            throw new Error(`Login failed: ${response.statusText}`);
        }

        const data = await response.json();

        // Store user data in localStorage for persistence
        localStorage.setItem('redline_user', JSON.stringify(data));

        return data;
    } catch (error) {
        console.error('Login error:', error);
        throw error;
    }
}

/**
 * Logout current user
 */
export function logout() {
    // Clear user data from localStorage
    localStorage.removeItem('redline_user');

    // If backend has a logout endpoint, call it here
    // fetch(`${API_BASE_URL}/api/auth/logout`, { method: 'POST', credentials: 'include' });
}

/**
 * Get current user from localStorage
 * @returns {Promise<Object|null>} User data or null
 */
export async function getCurrentUser() {
    try {
        const userStr = localStorage.getItem('redline_user');
        if (!userStr) {
            return null;
        }

        const user = JSON.parse(userStr);

        // Optional: Verify with backend that session is still valid
        // const response = await fetch(`${API_BASE_URL}/api/auth/me`, { credentials: 'include' });
        // if (!response.ok) {
        //   logout();
        //   return null;
        // }

        return user;
    } catch (error) {
        console.error('Get current user error:', error);
        logout(); // Clear invalid data
        return null;
    }
}

/**
 * Register a new user (if registration is enabled)
 * @param {Object} formData - Full registration form data
 * @returns {Promise<Object>} Registration result
 */
export async function register(formData) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Registration failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Registration error:', error);
        throw error;
    }
}

/**
 * Admin: Get all registration applications
 * @param {string} status - Filter by status (pending/approved/rejected)
 * @returns {Promise<Object>} Applications list
 */
export async function getApplications(status = null) {
    try {
        const url = status
            ? `${API_BASE_URL}/api/admin/applications?status=${status}`
            : `${API_BASE_URL}/api/admin/applications`;

        const response = await fetch(url, {
            credentials: 'include',
        });

        if (!response.ok) {
            throw new Error('Failed to fetch applications');
        }

        return await response.json();
    } catch (error) {
        console.error('Get applications error:', error);
        throw error;
    }
}

/**
 * Admin: Approve an application
 * @param {number} applicationId 
 * @param {string} adminNotes 
 * @returns {Promise<Object>} Result
 */
export async function approveApplication(applicationId, adminNotes = '') {
    try {
        const response = await fetch(`${API_BASE_URL}/api/admin/applications/${applicationId}/approve`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ adminNotes }),
            credentials: 'include',
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Approval failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Approve application error:', error);
        throw error;
    }
}

/**
 * Admin: Reject an application
 * @param {number} applicationId 
 * @param {string} reason 
 * @param {string} adminNotes 
 * @returns {Promise<Object>} Result
 */
export async function rejectApplication(applicationId, reason = '', adminNotes = '') {
    try {
        const response = await fetch(`${API_BASE_URL}/api/admin/applications/${applicationId}/reject`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ reason, adminNotes }),
            credentials: 'include',
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Rejection failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Reject application error:', error);
        throw error;
    }
}
