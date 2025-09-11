import express from 'express';
import Case from '../module/case.module.js';
import User from '../module/user.module.js';
import CommonUser from '../module/commonUser.module.js';
import LawyerProfile from '../module/lawyer.module.js';

const router = express.Router();

router.get('/', (req, res) => {
    res.send('Case Router is working');
});

router.post('/create', async (req, res) => {
    try {
        const { title, description, legal_category, client_id, lawyer_id } = req.body;

        // Basic validation
        if (!title || !description || !legal_category || !client_id || !lawyer_id) {
            return res.status(400).json({
                success: false,
                message: "All fields are required: title, description, legal_category, client_id, lawyer_id"
            });
        }

        const commonUser = await CommonUser.findById(client_id);
        if (!commonUser) {
            return res.status(404).json({
                success: false,
                message: "Client not found"
            });
        }
        
        const lawyerProfile = await LawyerProfile.findById(lawyer_id);
        if(!lawyerProfile){
            return res.status(404).json({
                success: false,
                message: "Lawyer not found"
            }); 
        }

        // Create new case
        const newCase = new Case({
            title,
            description,
            legal_category,
            client_id,
            lawyer_id,
            status: 'ongoing',
            process_updates: [{ update_text: 'Case created',updated_by: lawyer_id, timestamp: new Date() }]
        });
        
        const savedCase = await newCase.save();

        // Update CommonUser's cases array

        commonUser.cases.push({
            case_id: savedCase._id,
            lawyer_id: lawyerProfile._id,
            status: 'ongoing',
            last_updated: new Date()
        });

        await commonUser.save();

        lawyerProfile.cases.push({
            case_id: savedCase._id,
            client_id: commonUser._id,
            status: 'ongoing',
            last_update: new Date()
        });

        await lawyerProfile.save();

        res.status(201).json({
            success: true,
            message: "Case created successfully",
            case: savedCase
        });
    } catch (error) {
        console.log("Case creation error: ", error);
        res.status(500).json({
            success: false,
            message: "Server Error",
            error: error.message
        });
    }
});

export default router;