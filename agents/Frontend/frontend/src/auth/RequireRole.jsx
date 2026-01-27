import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
// ZMIANA: Importujemy useAuth z AuthContext
import { useAuth } from './AuthContext';

const RequireRole = ({ children, allowedRoles }) => {
    const { user, loading } = useAuth();
    const location = useLocation();

    if (loading) {
        // Prosty loader, żeby nie migało
        return <div className="min-h-screen bg-black flex items-center justify-center text-red-500 font-mono animate-pulse">VERIFYING CLEARANCE...</div>;
    }

    if (!user) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    // ROOT has access to everything
    // Otherwise check if user's role is in allowedRoles
    const userRole = user.role?.toUpperCase();
    const hasAccess = userRole === 'ROOT' ||
        !allowedRoles ||
        allowedRoles.map(r => r.toUpperCase()).includes(userRole);

    if (!hasAccess) {
        // If user doesn't have permissions -> Return to home
        return <Navigate to="/" replace />;
    }

    return children;
};

export default RequireRole;