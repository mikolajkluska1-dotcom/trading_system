import { useContext } from 'react';
import { AuthContext } from './AuthContext';

// To jest Named Export (w klamerkach), którego szuka login.jsx
export const useAuth = () => {
    const context = useContext(AuthContext);

    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }

    return context;
};

// Dodajemy też default export na wszelki wypadek, żeby inne pliki się nie wysypały
export default useAuth;